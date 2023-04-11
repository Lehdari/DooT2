//
// Project: DooT2
// File: AutoEncoderModel2.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "AutoEncoderModel2.hpp"
#include "SequenceStorage.hpp"
#include "Constants.hpp"

#include <opencv2/core/mat.hpp> // TODO temp
#include <opencv2/highgui.hpp> // TODO temp

#include <filesystem>
#include <random>


static constexpr double     learningRate            = 1.0e-4; // TODO
static constexpr int64_t    nTrainingIterations     = 6*64;


using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;


namespace {

    INLINE torch::Tensor yuvLoss(const torch::Tensor& target, const torch::Tensor& pred) {
        return torch::mean(torch::abs(target-pred), {0, 2, 3}) // reduce only batch and spatial dimensions, preserve channels
            .dot(torch::tensor({1.8f, 1.1f, 1.1f}, // higher weight on Y channel
            TensorOptions().device(target.device())));
    }

    void imShowYUV(const std::string& windowName, cv::Mat& image);

    // Loss function for YUV images
    using ImageLossTensors = std::tuple<Tensor, Tensor, Tensor, Tensor>;
    inline ImageLossTensors imageLoss(const torch::Tensor& target, const torch::Tensor& pred)
    {
        using namespace torch::indexing;

        // Pixel-space gradients
        torch::Tensor targetGradX =
            target.index({Slice(), Slice(), Slice(), Slice(1, None)}) -
            target.index({Slice(), Slice(), Slice(), Slice(None, -1)});
        torch::Tensor targetGradY =
            target.index({Slice(), Slice(), Slice(1, None), Slice()}) -
            target.index({Slice(), Slice(), Slice(None, -1), Slice()});
        torch::Tensor predGradX =
            pred.index({Slice(), Slice(), Slice(), Slice(1, None)}) -
            pred.index({Slice(), Slice(), Slice(), Slice(None, -1)});
        torch::Tensor predGradY =
            pred.index({Slice(), Slice(), Slice(1, None), Slice()}) -
            pred.index({Slice(), Slice(), Slice(None, -1), Slice()});

        // Pixel-space laplacian
        static float laplacianKernelData[9] = {
            -0.5,    -1.0,   -0.5,
            -1.0,   6.0,    -1.0,
            -0.5,    -1.0,   -0.5
        };
        static torch::Tensor laplacianKernel = torch::from_blob(laplacianKernelData, {3,3})
            .repeat({3,1,1,1}).to(target.device());
        torch::Tensor targetLaplacian = tf::conv2d(target, laplacianKernel, tf::Conv2dFuncOptions().padding(1).groups(3));
        torch::Tensor predLaplacian = tf::conv2d(pred, laplacianKernel, tf::Conv2dFuncOptions().padding(1).groups(3));

        return {
            yuvLoss(target, pred),
            yuvLoss(targetGradX, predGradX),
            yuvLoss(targetGradY, predGradY),
            yuvLoss(targetLaplacian, predLaplacian)
        };
    }

    void imShowYUV(const std::string& windowName, cv::Mat& image) {
        Image<float> imageYUV(image.cols, image.rows, ImageFormat::YUV, reinterpret_cast<float*>(image.data));
        Image<float> imageBGRA;
        convertImage(imageYUV, imageBGRA, ImageFormat::BGRA);
        cv::Mat matBGRA(image.rows, image.cols, CV_32FC4, const_cast<float*>(imageBGRA.data()));

        cv::imshow(windowName, matBGRA);
    }

}


AutoEncoderModel2::AutoEncoderModel2() :
    _optimizer          ({
        _frameEncoder->parameters(),
        _frameDecoder->parameters()},
        torch::optim::AdamWOptions(learningRate).betas({0.99, 0.999}).weight_decay(1.0e-7)),
    _frameLossSmooth    (-1.0),
    _skipLevel          (7.0) // 7.0
{
    using namespace doot2;

    // Load frame encoder
    if (fs::exists(frameEncoderFilename)) {
        printf("Loading frame encoder model from %s\n", frameEncoderFilename); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameEncoderFilename);
        _frameEncoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame encoder model.\n", frameEncoderFilename); // TODO logging
    }

    // Load frame decoder
    if (fs::exists(frameDecoderFilename)) {
        printf("Loading frame decoder model from %s\n", frameDecoderFilename); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameDecoderFilename);
        _frameDecoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame decoder model.\n", frameDecoderFilename); // TODO logging
    }
}

void AutoEncoderModel2::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // Check CUDA availability
    torch::Device device{torch::kCPU};
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU); // shorthand for above

    cv::Mat imageIn1(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageOut(480, 640, CV_32FC3); // TODO temp

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);

    // Load the whole storage's pixel data to the GPU
    torch::Tensor pixelDataIn_ = storage.mapPixelData();
    torch::Tensor pixelDataIn = pixelDataIn_.to(device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3});

    Tensor flowBase = tf::affine_grid(torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        TensorOptions().device(device)).broadcast_to({doot2::batchSize, 2, 3}),
        {doot2::batchSize, 2, 480, 640});

    // Required for latent space normalization
    torch::Tensor encodingZeros = torch::zeros({doot2::encodingLength}, TensorOptions().device(device));
    torch::Tensor encodingOnes = torch::ones({doot2::encodingLength}, TensorOptions().device(device));

    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();

    const auto sequenceLength = storage.settings().length;
    const int64_t optimizationInterval = 8; // for this many frames gradients will be accumulated before update
    for (int64_t ti=0; ti<nTrainingIterations; ++ti) {
        // frame (time point) to use this iteration
        int64_t t = rnd() % (sequenceLength);

        // Pick random sequence to display
        size_t displaySeqId = rnd() % doot2::batchSize;

        // ID of the frame (in sequence) to be used in the training batch
        torch::Tensor batchIn1 = pixelDataIn.index({(int)t});

        // Forward passes
        auto [encoding1, s0, s1, s2, s3, s4, s5, s6] = _frameEncoder->forward(batchIn1); // image t encode

        // Frame decode
        auto [batchOut, levelMatchLoss] = _frameDecoder->forward(encoding1, s0, s1, s2, s3, s4, s5, s6, _skipLevel);

        // Frame losses
        auto [frameLoss, xGradLoss, yGradLoss, laplacianLoss] = imageLoss(batchIn1, batchOut);

        // Encoding mean/variance loss
        torch::Tensor encodingMean = torch::mean(encoding1, {0}); // mean across the batch dimension
        torch::Tensor encodingVar = torch::var(encoding1, 0); // variance across the batch dimension
        torch::Tensor encodingMeanLoss = 1.0e-6f * torch::sum(torch::square(encodingMean - encodingZeros));
        torch::Tensor encodingVarLoss = 1.0e-6f * torch::sum(torch::square(encodingVar - encodingOnes));

        // Total frame loss
        constexpr double xGradLossWeight = 4.0;
        constexpr double yGradLossWeight = 5.0;
        constexpr double laplacianLossWeight = 0.8;
        torch::Tensor totalFrameLoss =
            frameLoss
            + xGradLossWeight * xGradLoss
            + yGradLossWeight * yGradLoss;
            + laplacianLossWeight * laplacianLoss;

        // Total loss
        constexpr double levelMatchLossWeight = 10.0;
        torch::Tensor loss =
            totalFrameLoss
            + encodingMeanLoss
            + encodingVarLoss
            + levelMatchLossWeight * levelMatchLoss;

        // Backward pass
        loss.backward();

        // Smooth loss
        if (_frameLossSmooth < 0.0)
            _frameLossSmooth = totalFrameLoss.item<double>();
        else
            _frameLossSmooth = _frameLossSmooth*0.99 + totalFrameLoss.item<double>()*0.01;

        // Adjust skip level
        constexpr double targetLoss = 0.25;
        // P control
        constexpr double skipLevelControlPFactorPositive = 0.01;
        constexpr double skipLevelControlPFactorNegative = 0.001; // raise skipLevel slower
        double skipLevelControlP = (targetLoss-_frameLossSmooth);
        if (skipLevelControlP > 0.0)
            skipLevelControlP *= skipLevelControlPFactorPositive;
        else
            skipLevelControlP *= skipLevelControlPFactorNegative;
        _skipLevel -= skipLevelControlP;
        // limit to sensible domain
        _skipLevel = std::clamp(_skipLevel, -1.0, 7.0);

        // Apply gradients
        if (ti % optimizationInterval == optimizationInterval-1) {
            _optimizer.step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();

            // Display
            torch::Tensor batchIn1CPU = batchIn1.to(torch::kCPU);
            torch::Tensor outputCPU = batchOut.to(torch::kCPU);
            for (int j = 0; j < 480; ++j) {
                for (int i = 0; i < 640; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        imageIn1.ptr<float>(j)[i * 3 + c] = batchIn1CPU.data_ptr<float>()
                            [displaySeqId * 480 * 640 * 3 + j * 640 * 3 + i * 3 + c];
                        imageOut.ptr<float>(j)[i * 3 + c] = outputCPU.data_ptr<float>()
                            [displaySeqId * 3 * 480 * 640 + c * 480 * 640 + j * 640 + i];
                    }
                }
            }
            imShowYUV("Input 1", imageIn1);
            imShowYUV("Prediction 1", imageOut);
            cv::waitKey(1);
        }

        // Print loss
        printf(
            "\r[%2ld/%2ld][%3ld/%3ld] loss: %9.6f totalFrameLoss: %9.6f frameLoss: %9.6f xGradLoss: %9.6f yGradLoss: %9.6f laplacianLoss: %9.6f levelMatchLoss: %9.6f frameLossSmooth: %9.6f skipLevel: %9.6f",// encMean: [%9.6f %9.6f] encVar: [%9.6f %9.6f]",
            ti, nTrainingIterations, t, sequenceLength,
            loss.item<float>(), totalFrameLoss.item<float>(), frameLoss.item<float>(),
            xGradLoss.item<float>() * xGradLossWeight, yGradLoss.item<float>() * yGradLossWeight,
            laplacianLoss.item<float>() * laplacianLossWeight,
            levelMatchLoss.item<float>() * levelMatchLossWeight,
            _frameLossSmooth, _skipLevel
//            encodingMean.min().item<float>(), encodingMean.max().item<float>(),
//            encodingVar.min().item<float>(), encodingVar.max().item<float>()
        );
        fflush(stdout);
    }
    printf("\n");

    // Save models
    {
        printf("Saving frame encoder model to %s\n", doot2::frameEncoderFilename);
        serialize::OutputArchive outputArchive;
        _frameEncoder->save(outputArchive);
        outputArchive.save_to(doot2::frameEncoderFilename);
    }
    {
        printf("Saving frame decoder model to %s\n", doot2::frameDecoderFilename);
        serialize::OutputArchive outputArchive;
        _frameDecoder->save(outputArchive);
        outputArchive.save_to(doot2::frameDecoderFilename);
    }
}

void AutoEncoderModel2::infer(const TensorVector& input, TensorVector& output)
{
    // TODO
}

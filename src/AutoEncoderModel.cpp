//
// Project: DooT2
// File: AutoEncoderModel.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "AutoEncoderModel.hpp"
#include "SequenceStorage.hpp"
#include "Constants.hpp"

#include <opencv2/core/mat.hpp> // TODO temp
#include <opencv2/highgui.hpp> // TODO temp

#include <filesystem>
#include <random>


static constexpr double     learningRate            = 1.0e-3; // TODO
static constexpr int64_t    nTrainingIterations     = 4*64;


using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;


#define FLOW 1
#define DOUBLE_EDEC 1


namespace {

    INLINE torch::Tensor yuvLoss(const torch::Tensor& target, const torch::Tensor& pred) {
        return torch::mean(torch::abs(target-pred), {0, 2, 3}) // reduce only batch and spatial dimensions, preserve channels
            .dot(torch::tensor({2.0f, 1.0f, 1.0f}, // higher weight on Y channel
            TensorOptions().device(target.device())));
    }

    // Loss function for YUV images
    inline torch::Tensor imageLoss(const torch::Tensor& target, const torch::Tensor& pred, float gradWeight = 2.0f)
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

        return
            yuvLoss(target, pred) +
            gradWeight * yuvLoss(targetGradX, predGradX) +
            gradWeight * yuvLoss(targetGradY, predGradY);
    }

    void imShowYUV(const std::string& windowName, cv::Mat& image) {
        Image<float> imageYUV(image.cols, image.rows, ImageFormat::YUV, reinterpret_cast<float*>(image.data));
        Image<float> imageBGRA;
        convertImage(imageYUV, imageBGRA, ImageFormat::BGRA);
        cv::Mat matBGRA(image.rows, image.cols, CV_32FC4, const_cast<float*>(imageBGRA.data()));

        cv::imshow(windowName, matBGRA);
    }

}


AutoEncoderModel::AutoEncoderModel() :
    _optimizer          ({
        _frameEncoder->parameters(),
        _frameDecoder->parameters(),
        _flowDecoder->parameters()},
        torch::optim::AdamWOptions(learningRate).betas({0.9, 0.999}).weight_decay(0.001)),
    _trainingFinished   (true)
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

    // Load flow decoder
    if (fs::exists(flowDecoderFilename)) {
        printf("Loading frame decoder model from %s\n", flowDecoderFilename); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(flowDecoderFilename);
        _flowDecoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new flow decoder model.\n", flowDecoderFilename); // TODO logging
    }
}

void AutoEncoderModel::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // Return immediately in case there's previous training by another thread already running
    if (!_trainingFinished)
        return;

    _trainingFinished = false;
    std::lock_guard<std::mutex> lock(_trainingMutex);

    // Check CUDA availability
    torch::Device device{torch::kCPU};
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU); // shorthand for above

    cv::Mat imageIn1(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageIn1Scaled1(240, 320, CV_32FC3); // TODO temp
    cv::Mat imageIn1Scaled2(120, 160, CV_32FC3); // TODO temp
    cv::Mat imageIn2(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageOut(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageOut2(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageOutScaled1(240, 320, CV_32FC3); // TODO temp
    cv::Mat imageOutScaled2(120, 160, CV_32FC3); // TODO temp
    cv::Mat imageFlowForward(480, 640, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f)); // TODO temp
    cv::Mat imageFlowForwardMapped(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageFlowForwardDiff(480, 640, CV_32FC1); // TODO temp
    cv::Mat imageFlowBackward(480, 640, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f)); // TODO temp
    cv::Mat imageFlowBackwardMapped(480, 640, CV_32FC3); // TODO temp
    cv::Mat imageFlowBackwardDiff(480, 640, CV_32FC1); // TODO temp
    cv::Mat imageInGradX(480, 639, CV_32FC3); // TODO temp
    cv::Mat imageInGradY(479, 640, CV_32FC3); // TODO temp

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);
    _flowDecoder->to(device);

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
    _flowDecoder->zero_grad();

    const auto sequenceLength = storage.settings().length;
    const int64_t optimizationInterval = 8; // for this many frames gradients will be accumulated before update
    for (int64_t ti=0; ti<nTrainingIterations; ++ti) {
        // frame (time point) to use this iteration
        int64_t t = rnd() % (sequenceLength-1);

        // Pick random sequence to display
        size_t displaySeqId = rnd() % doot2::batchSize;

        // ID of the frame (in sequence) to be used in the training batch
        torch::Tensor batchIn1 = pixelDataIn.index({(int)t});
        torch::Tensor batchIn2 = pixelDataIn.index({(int)t+1});
        torch::Tensor batchIn1Scaled1 = tf::interpolate(batchIn1,
            tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
        torch::Tensor batchIn1Scaled2 = tf::interpolate(batchIn1,
            tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));

        // Forward passes
        torch::Tensor encoding1 = _frameEncoder->forward(batchIn1); // image t encode
        torch::Tensor encoding2 = _frameEncoder->forward(batchIn2); // image t+1 encode
#if FLOW
        // Flow from frame t to t+1
        torch::Tensor flowForward = _flowDecoder->forward(encoding2-encoding1);
        torch::Tensor flowFrameForward = tf::grid_sample(batchIn1, flowBase+flowForward,
            tf::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kBorder)
            .align_corners(true));

        // Flow from frame t+1 to t
        torch::Tensor flowBackward = _flowDecoder->forward(encoding1-encoding2);
        torch::Tensor flowFrameBackward = tf::grid_sample(batchIn2, flowBase+flowBackward,
            tf::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true));
#endif
        // Frame decode
        auto tupleOut = _frameDecoder->forward(encoding1);
        auto& batchOut = std::get<0>(tupleOut);
        auto& batchOutScaled1 = std::get<1>(tupleOut);
        auto& batchOutScaled2 = std::get<2>(tupleOut);

#if DOUBLE_EDEC
        // Double encode-decode
        torch::Tensor encoding1double = _frameEncoder->forward(batchOut);
        auto tupleOut2 = _frameDecoder->forward(encoding1double);
        auto& batchOut2 = std::get<0>(tupleOut2);
        auto& batchOut2Scaled1 = std::get<1>(tupleOut2);
        auto& batchOut2Scaled2 = std::get<2>(tupleOut2);
#endif

        // Frame losses
        torch::Tensor frameLoss =
            imageLoss(batchIn1, batchOut) +
            imageLoss(batchIn1Scaled1, batchOutScaled1) +
            imageLoss(batchIn1Scaled2, batchOutScaled2);
#if DOUBLE_EDEC
        torch::Tensor frameLossDoubleEdec =
            imageLoss(batchIn1, batchOut2) +
            imageLoss(batchIn1Scaled1, batchOut2Scaled1) +
            imageLoss(batchIn1Scaled2, batchOut2Scaled2);
#endif
#if FLOW
        // Flow loss
        torch::Tensor flowForwardDiff = torch::abs(
            batchIn2.index({Slice(), 0, "..."}) -
            flowFrameForward.index({Slice(), 0, "..."}));
        torch::Tensor flowBackwardDiff = torch::abs(
            batchIn1.index({Slice(), 0, "..."}) -
            flowFrameBackward.index({Slice(), 0, "..."}));
        torch::Tensor flowForwardLoss = 5.0f * torch::mean(flowForwardDiff);
        torch::Tensor flowBackwardLoss = 5.0f * torch::mean(flowBackwardDiff);
#endif
        // Encoding mean/variance loss
        torch::Tensor encodingMean = torch::mean(encoding1, {0}); // mean across the batch dimension
        torch::Tensor encodingVar = torch::var(encoding1, 0); // variance across the batch dimension
        torch::Tensor encodingMeanLoss = 1.0e-6f * torch::sum(torch::square(encodingMean - encodingZeros));
        torch::Tensor encodingVarLoss = 1.0e-6f * torch::sum(torch::square(encodingVar - encodingOnes));
#if DOUBLE_EDEC
        // Double encoding loss
        torch::Tensor encoding1detached = encoding1.detach(); // prevent gradient flowing back through encoding1 so that the double edec loss won't pull it towards the worse double encoding
        torch::Tensor doubleEncodingLoss = 1.0e-4f * torch::mse_loss(encoding1detached, encoding1double);
#endif

        // Total loss
        torch::Tensor loss =
            frameLoss
#if FLOW
            + flowForwardLoss
            + flowBackwardLoss
#endif
#if DOUBLE_EDEC
            + frameLossDoubleEdec
            + doubleEncodingLoss
#endif
            + encodingMeanLoss
            + encodingVarLoss;

        // Backward pass
        loss.backward();

        // Apply gradients
        if (ti % optimizationInterval == optimizationInterval-1) {
            _optimizer.step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();
            _flowDecoder->zero_grad();

            // Display
            torch::Tensor batchIn1CPU = batchIn1.to(torch::kCPU);
            torch::Tensor batchIn1Scaled1CPU = batchIn1Scaled1.to(torch::kCPU);
            torch::Tensor batchIn1Scaled2CPU = batchIn1Scaled2.to(torch::kCPU);
            torch::Tensor batchIn2CPU = batchIn2.to(torch::kCPU);
#if FLOW
            torch::Tensor flowForwardCPU = flowForward.to(torch::kCPU) * 2.0f + 0.5f;
            torch::Tensor flowFrameForwardCPU = flowFrameForward.to(torch::kCPU);
            torch::Tensor flowForwardDiffCPU = flowForwardDiff.contiguous().to(torch::kCPU);
            torch::Tensor flowBackwardCPU = flowBackward.to(torch::kCPU) * 2.0f + 0.5f;
            torch::Tensor flowFrameBackwardCPU = flowFrameBackward.to(torch::kCPU);
            torch::Tensor flowBackwardDiffCPU = flowBackwardDiff.contiguous().to(torch::kCPU);
#endif
            torch::Tensor outputCPU = batchOut.to(torch::kCPU);
#if DOUBLE_EDEC
            torch::Tensor output2CPU = batchOut2.to(torch::kCPU);
#endif
            torch::Tensor batchOutScaled1CPU = batchOutScaled1.to(torch::kCPU);
            torch::Tensor batchOutScaled2CPU = batchOutScaled2.to(torch::kCPU);
            for (int j = 0; j < 480; ++j) {
                for (int i = 0; i < 640; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        imageIn1.ptr<float>(j)[i * 3 + c] = batchIn1CPU.data_ptr<float>()
                            [displaySeqId * 480 * 640 * 3 + j * 640 * 3 + i * 3 + c];
                        imageIn2.ptr<float>(j)[i * 3 + c] = batchIn2CPU.data_ptr<float>()
                            [displaySeqId * 480 * 640 * 3 + j * 640 * 3 + i * 3 + c];
#if FLOW
                        imageFlowForwardMapped.ptr<float>(j)[i * 3 + c] = flowFrameForwardCPU.data_ptr<float>()
                            [displaySeqId * 3 * 480 * 640 + c * 480 * 640 + j * 640 + i];
                        imageFlowBackwardMapped.ptr<float>(j)[i * 3 + c] = flowFrameBackwardCPU.data_ptr<float>()
                            [displaySeqId * 3 * 480 * 640 + c * 480 * 640 + j * 640 + i];
#endif
                        imageOut.ptr<float>(j)[i * 3 + c] = outputCPU.data_ptr<float>()
                            [displaySeqId * 3 * 480 * 640 + c * 480 * 640 + j * 640 + i];
#if DOUBLE_EDEC
                        imageOut2.ptr<float>(j)[i * 3 + c] = output2CPU.data_ptr<float>()
                            [displaySeqId * 3 * 480 * 640 + c * 480 * 640 + j * 640 + i];
#endif
                    }
#if FLOW
                    for (int c = 0; c < 2; ++c) {
                        imageFlowForward.ptr<float>(j)[i * 3 + c] = flowForwardCPU.data_ptr<float>()
                            [displaySeqId * 2 * 480 * 640 + c * 480 * 640 + j * 640 + i];
                        imageFlowBackward.ptr<float>(j)[i * 3 + c] = flowBackwardCPU.data_ptr<float>()
                            [displaySeqId * 2 * 480 * 640 + c * 480 * 640 + j * 640 + i];
                    }
                    imageFlowForwardDiff.ptr<float>(j)[i] = flowForwardDiffCPU.data_ptr<float>()
                        [displaySeqId * 480 * 640 + j * 640 + i];
                    imageFlowBackwardDiff.ptr<float>(j)[i] = flowBackwardDiffCPU.data_ptr<float>()
                        [displaySeqId * 480 * 640 + j * 640 + i];
#endif
                }
            }
            for (int j = 0; j < 240; ++j) {
                for (int i = 0; i < 320; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        imageIn1Scaled1.ptr<float>(j)[i * 3 + c] = batchIn1Scaled1CPU.data_ptr<float>()
                            [displaySeqId * 240 * 320 * 3 + j * 320 * 3 + i * 3 + c];
                        imageOutScaled1.ptr<float>(j)[i * 3 + c] = batchOutScaled1CPU.data_ptr<float>()
                            [displaySeqId * 3 * 240 * 320 + c * 240 * 320 + j * 320 + i];
                    }
                }
            }
            for (int j = 0; j < 120; ++j) {
                for (int i = 0; i < 160; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        imageIn1Scaled2.ptr<float>(j)[i * 3 + c] = batchIn1Scaled2CPU.data_ptr<float>()
                            [displaySeqId * 120 * 160 * 3 + j * 160 * 3 + i * 3 + c];
                        imageOutScaled2.ptr<float>(j)[i * 3 + c] = batchOutScaled2CPU.data_ptr<float>()
                            [displaySeqId * 3 * 120 * 160 + c * 120 * 160 + j * 160 + i];
                    }
                }
            }
            imShowYUV("Input 1", imageIn1);
            imShowYUV("Input 1 Scaled 1", imageIn1Scaled1);
            imShowYUV("Input 1 Scaled 2", imageIn1Scaled2);
            imShowYUV("Input 2", imageIn2);
#if FLOW
            cv::imshow("Forward Flow", imageFlowForward);
            imShowYUV("Forward Flow Mapped", imageFlowForwardMapped);
            cv::imshow("Forward Flow Diff", imageFlowForwardDiff);
            cv::imshow("Backward Flow", imageFlowBackward);
            imShowYUV("Backward Flow Mapped", imageFlowBackwardMapped);
            cv::imshow("Backward Flow Diff", imageFlowBackwardDiff);
#endif
            imShowYUV("Prediction 1", imageOut);
#if DOUBLE_EDEC
            imShowYUV("Prediction 1, Double EDEC", imageOut2);
#endif
            imShowYUV("Prediction 1 Scaled 1", imageOutScaled1);
            imShowYUV("Prediction 1 Scaled 2", imageOutScaled2);
            cv::waitKey(1);
        }

        // Print loss
        printf(
            "\r[%2ld/%2ld][%3ld/%3ld] loss: %9.6f frameLoss: %9.6f flowForwardLoss: %9.6f flowBackwardLoss: %9.6f frameLossDoubleEdec: %9.6f doubleEdecLoss: %9.6f encMean: [%9.6f %9.6f] encVar: [%9.6f %9.6f]",
            ti, nTrainingIterations, t, sequenceLength,
            loss.item<float>(), frameLoss.item<float>(),
#if FLOW
            flowForwardLoss.item<float>(), flowBackwardLoss.item<float>(),
#else
            0.0, 0.0,
#endif
#if DOUBLE_EDEC
            frameLossDoubleEdec.item<float>(), doubleEncodingLoss.item<float>(),
#else
            0.0, 0.0,
#endif
            encodingMean.min().item<float>(), encodingMean.max().item<float>(),
            encodingVar.min().item<float>(), encodingVar.max().item<float>());
        fflush(stdout);
    }

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
    {
        printf("Saving flow decoder model to %s\n", doot2::flowDecoderFilename);
        serialize::OutputArchive outputArchive;
        _flowDecoder->save(outputArchive);
        outputArchive.save_to(doot2::flowDecoderFilename);
    }

    _trainingFinished = true;
}

void AutoEncoderModel::infer(const TensorVector& input, TensorVector& output)
{
    // TODO
}

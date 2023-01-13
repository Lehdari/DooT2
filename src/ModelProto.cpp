//
// Project: DooT2
// File: ModelProto.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ModelProto.hpp"
#include "SequenceStorage.hpp"

#include <opencv2/core/mat.hpp> // TODO temp
#include <opencv2/highgui.hpp> // TODO temp

#include <filesystem>


static constexpr int        batchSize               = 16; // TODO move somewhere sensible
static constexpr double     learningRate            = 1.0e-4; // TODO
static constexpr int64_t    nTrainingEpochs         = 8;
static constexpr char       frameEncoderFilename[]  {"frame_encoder.pt"};
static constexpr char       frameDecoderFilename[]  {"frame_decoder.pt"};
static constexpr char       flowDecoderFilename[]   {"flow_decoder.pt"};


using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;


ModelProto::ModelProto() :
    _optimizer          ({
        _frameEncoder->parameters(),
        _frameDecoder->parameters(),
        _flowDecoder->parameters()},
        torch::optim::AdamOptions(learningRate).betas({0.9, 0.999})),
    _trainingFinished   (true)
{
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

void ModelProto::train(SequenceStorage&& storage)
{
    using namespace torch::indexing;

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

    cv::Mat imageIn1(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageIn2(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageOut(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageFlowForward(480, 640, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f)); // TODO temp
    cv::Mat imageFlowForwardMapped(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageFlowForwardDiff(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageFlowBackward(480, 640, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f)); // TODO temp
    cv::Mat imageFlowBackwardMapped(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageFlowBackwardDiff(480, 640, CV_32FC4); // TODO temp

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);
    _flowDecoder->to(device);

    // Load the whole storage's pixel data to the GPU
    torch::Tensor pixelDataIn_ = storage.mapPixelData();
    torch::Tensor pixelDataIn = pixelDataIn_.to(device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3});

    Tensor flowBase = tf::affine_grid(torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        TensorOptions().device(device)).broadcast_to({batchSize, 2, 3}),
        {batchSize, 2, 480, 640});

    const auto sequenceLength = storage.settings().length;
    for (int64_t epoch=0; epoch<nTrainingEpochs; ++epoch) {
        // Accumulate gradients over a sequence
        _frameEncoder->zero_grad();
        _frameDecoder->zero_grad();
        _flowDecoder->zero_grad();

        for (int64_t t=0; t<sequenceLength-1; ++t) {
            // ID of the frame (in sequence) to be used in the training batch
            torch::Tensor batchIn1 = pixelDataIn.index({(int)t});
            torch::Tensor batchIn2 = pixelDataIn.index({(int)t+1});

            // Forward and backward passes
            torch::Tensor encoding1 = _frameEncoder->forward(batchIn1); // image t encode
            torch::Tensor encoding2 = _frameEncoder->forward(batchIn2); // image t+1 encode

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

            // Frame decode and loss
            torch::Tensor batchOut = _frameDecoder->forward(encoding1);
            torch::Tensor frameLoss = torch::l1_loss(batchIn1, batchOut) + torch::mse_loss(batchIn1, batchOut);

            // Flow loss
            torch::Tensor flowForwardDiff = torch::abs(batchIn2-flowFrameForward);
            torch::Tensor flowBackwardDiff = torch::abs(batchIn1-flowFrameBackward);
            torch::Tensor flowForwardLoss = torch::mean(flowForwardDiff);
            torch::Tensor flowBackwardLoss = torch::mean(flowBackwardDiff);

            // Total loss
            torch::Tensor loss = frameLoss + flowForwardLoss + flowBackwardLoss;
            loss.backward();

            // Cycle through displayed sequences
            size_t displayFrameId = epoch;
            torch::Tensor batchIn1CPU = batchIn1.to(torch::kCPU);
            torch::Tensor batchIn2CPU = batchIn2.to(torch::kCPU);
            torch::Tensor flowForwardCPU = flowForward.to(torch::kCPU)*2.0f + 0.5f;
            torch::Tensor flowFrameForwardCPU = flowFrameForward.to(torch::kCPU);
            torch::Tensor flowForwardDiffCPU = flowForwardDiff.to(torch::kCPU);
            torch::Tensor flowBackwardCPU = flowBackward.to(torch::kCPU)*2.0f + 0.5f;
            torch::Tensor flowFrameBackwardCPU = flowFrameBackward.to(torch::kCPU);
            torch::Tensor flowBackwardDiffCPU = flowBackwardDiff.to(torch::kCPU);
            torch::Tensor outputCPU = batchOut.to(torch::kCPU);
            for (int j=0; j<480; ++j) {
                for (int i=0; i<640; ++i) {
                    for (int c=0; c<4; ++c) {
                        imageIn1.ptr<float>(j)[i*4 + c] = batchIn1CPU.data_ptr<float>()
                            [displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                        imageIn2.ptr<float>(j)[i*4 + c] = batchIn2CPU.data_ptr<float>()
                            [displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                        imageFlowForwardMapped.ptr<float>(j)[i*4 + c] = flowFrameForwardCPU.data_ptr<float>()
                            [displayFrameId*4*480*640 + c*480*640 + j*640 + i];
                        imageFlowForwardDiff.ptr<float>(j)[i*4 + c] = flowForwardDiffCPU.data_ptr<float>()
                            [displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                        imageFlowBackwardMapped.ptr<float>(j)[i*4 + c] = flowFrameBackwardCPU.data_ptr<float>()
                            [displayFrameId*4*480*640 + c*480*640 + j*640 + i];
                        imageFlowBackwardDiff.ptr<float>(j)[i*4 + c] = flowBackwardDiffCPU.data_ptr<float>()
                            [displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                        imageOut.ptr<float>(j)[i*4 + c] = outputCPU.data_ptr<float>()
                            [displayFrameId*4*480*640 + c*480*640 + j*640 + i];
                    }
                    for (int c=0; c<2; ++c) {
                        imageFlowForward.ptr<float>(j)[i*3 + c] = flowForwardCPU.data_ptr<float>()
                            [displayFrameId*2*480*640 + c*480*640 + j*640 + i];
                        imageFlowBackward.ptr<float>(j)[i*3 + c] = flowBackwardCPU.data_ptr<float>()
                            [displayFrameId*2*480*640 + c*480*640 + j*640 + i];
                    }
                }
            }
            cv::imshow("Input 1", imageIn1);
            cv::imshow("Input 2", imageIn2);
            cv::imshow("Forward Flow", imageFlowForward);
            cv::imshow("Forward Flow Mapped", imageFlowForwardMapped);
            cv::imshow("Forward Flow Diff", imageFlowForwardDiff);
            cv::imshow("Backward Flow", imageFlowBackward);
            cv::imshow("Backward Flow Mapped", imageFlowBackwardMapped);
            cv::imshow("Backward Flow Diff", imageFlowBackwardDiff);
            cv::imshow("Prediction 1", imageOut);
            cv::waitKey(1);

            // TEMP encoding mean and variance
            torch::Tensor encodingMean = encoding1.mean();
            torch::Tensor encodingVar = encoding1.var();

            printf(
                "\r[%2ld/%2ld][%3ld/%3ld] loss: %9.6f frameLoss: %9.6f flowForwardLoss: %9.6f flowBackwardLoss: %9.6f encodingMean: %9.6f encodingVar: %9.6f",
                epoch, nTrainingEpochs, t, sequenceLength,
                loss.item<float>(), frameLoss.item<float>(),
                flowForwardLoss.item<float>(), flowBackwardLoss.item<float>(),
                encodingMean.item<float>(), encodingVar.item<float>());
            fflush(stdout);
        }

        // Apply accumulated gradients
        _optimizer.step();
    }

    // Save models
    {
        printf("Saving frame encoder model to %s\n", frameEncoderFilename);
        serialize::OutputArchive outputArchive;
        _frameEncoder->save(outputArchive);
        outputArchive.save_to(frameEncoderFilename);
    }
    {
        printf("Saving frame decoder model to %s\n", frameDecoderFilename);
        serialize::OutputArchive outputArchive;
        _frameDecoder->save(outputArchive);
        outputArchive.save_to(frameDecoderFilename);
    }
    {
        printf("Saving flow decoder model to %s\n", flowDecoderFilename);
        serialize::OutputArchive outputArchive;
        _flowDecoder->save(outputArchive);
        outputArchive.save_to(flowDecoderFilename);
    }

    _trainingFinished = true;
}

void ModelProto::trainAsync(SequenceStorage&& storage)
{
    if (_trainingThread.joinable())
        _trainingThread.join();
    _trainingThread = std::thread{&ModelProto::train, this, std::move(storage)};
}

bool ModelProto::trainingFinished() const noexcept
{
    return _trainingFinished;
}

void ModelProto::waitForTrainingFinish()
{
    if (_trainingFinished)
        return;

    if (_trainingThread.joinable())
        _trainingThread.join();
}

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


static constexpr int    batchSize           = 16; // TODO move somewhere sensible
static constexpr double learningRate        = 1.0e-3; // TODO
static constexpr int    nTrainingEpochs     = 1024;
static constexpr char   frameEncoderFilename[]  {"frame_encoder.pt"};
static constexpr char   frameDecoderFilename[]  {"frame_decoder.pt"};


using namespace torch;
namespace fs = std::filesystem;


ModelProto::ModelProto() :
    _optimizer          ({_frameEncoder->parameters(), _frameDecoder->parameters()}, torch::optim::AdamOptions(learningRate).betas({0.9, 0.999})),
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
}

void ModelProto::train(const SequenceStorage& storage)
{
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

    cv::Mat imageIn(480, 640, CV_32FC4); // TODO temp
    cv::Mat imageOut(480, 640, CV_32FC4); // TODO temp

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);

    torch::Tensor batchIn;
    for (int64_t epoch=0; epoch<nTrainingEpochs; ++epoch) {
        // ID of the frame (in sequence) to be used in the training batch
        size_t trainFrameId = epoch % storage.settings().length;
        batchIn = torch::from_blob(
            const_cast<float*>(storage[trainFrameId].frames->data()), // TODO very bad, create custom accessor for this
            {batchSize, 480, 640, 4}, torch::TensorOptions().device(torch::kCPU));
        auto* batchDataIn = batchIn.data_ptr<float>();

        torch::Tensor batchInGPU = batchIn.to(device);
        batchInGPU = batchInGPU.transpose(1, 3);
        batchInGPU = batchInGPU.transpose(2, 3);

        // Forward and backward passes
        _frameEncoder->zero_grad();
        _frameDecoder->zero_grad();
        torch::Tensor encoding = _frameEncoder->forward(batchInGPU); // image encode
        torch::Tensor batchOut = _frameDecoder->forward(encoding); // image decode
        torch::Tensor loss = torch::l1_loss(batchInGPU, batchOut) + torch::mse_loss(batchInGPU, batchOut);
        loss.backward();

        // Apply gradients
        _optimizer.step();

        // Cycle through displayed sequences
        size_t displayFrameId = (epoch/storage.settings().length)%batchSize;
        torch::Tensor outputCPU = batchOut.to(torch::kCPU);
        auto* batchDataOut = outputCPU.data_ptr<float>();
        for (int c=0; c<4; ++c) {
            for (int j=0; j<480; ++j) {
                for (int i=0; i<640; ++i) {
                    imageIn.ptr<float>(j)[i*4 + c] = batchDataIn[displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                    imageOut.ptr<float>(j)[i*4 + c] = batchDataOut[displayFrameId*480*640*4 + j*640*4 + i*4 + c];
                }
            }
        }
        cv::imshow("Target", imageIn);
        cv::imshow("Prediction", imageOut);
        cv::waitKey(1);

        printf(
            "\r[%2ld/%2ld][%3ld/%3ld] loss: %.6f",
            epoch,
            nTrainingEpochs,
            0,//++batch_index,
            0,//batches_per_epoch,
            loss.item<float>());
        fflush(stdout);
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

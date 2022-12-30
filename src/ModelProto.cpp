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


static constexpr int    batchSize           = 16; // TODO move somewhere sensible
static constexpr double learningRate        = 1.0e-3; // TODO
static constexpr int    nTrainingEpochs     = 1024;


ModelProto::ModelProto() :
    _optimizer  (_autoEncoder->parameters(), torch::optim::AdamOptions(learningRate).betas({0.9, 0.999}))
{
}

void ModelProto::train(const SequenceStorage& storage)
{
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
    _autoEncoder->to(device);

    torch::Tensor batchIn = torch::ones({batchSize, 4, 480, 640}, torch::kCPU);
    auto* batchDataIn = batchIn.data_ptr<float>();
    Image<float> frame;
    for (int64_t epoch=0; epoch<nTrainingEpochs; ++epoch) {
        // ID of the frame (in sequence) to be used in the training batch
        size_t trainFrameId = epoch%storage.size();

        // Transfer data into the input tensor
        for (int b=0; b<batchSize; ++b) {
            convertImage(storage[trainFrameId][b].bgraFrame, frame); // do inline conversion for now (TODO)
            // transpose the image and add it to the batch data vector
            for (int c=0; c<4; ++c) {
                for (int j=0; j<480; ++j) {
                    for (int i=0; i<640; ++i) {
                        batchDataIn[b*4*480*640 + c*480*640 + j*640 + i] = frame.data()[j*640*4 + i*4 + c];
                    }
                }
            }
        }
        torch::Tensor batchInGPU = batchIn.to(device);

        // Forward and backward passes
        _autoEncoder->zero_grad();
        torch::Tensor batchOut = _autoEncoder->forward(batchInGPU);
        torch::Tensor loss = torch::mse_loss(batchInGPU, batchOut);
        loss.backward();

        // Apply gradients
        _optimizer.step();

        // Cycle through displayed sequences
        size_t displayFrameId = (epoch/storage.size())%batchSize;
        torch::Tensor outputCPU = batchOut.to(torch::kCPU);
        auto* batchDataOut = outputCPU.data_ptr<float>();
        for (int c=0; c<4; ++c) {
            for (int j=0; j<480; ++j) {
                for (int i=0; i<640; ++i) {
                    imageIn.ptr<float>(j)[i*4 + c] = batchDataIn[displayFrameId*4*480*640 + c*480*640 + j*640 + i];
                    imageOut.ptr<float>(j)[i*4 + c] = batchDataOut[displayFrameId*4*480*640 + c*480*640 + j*640 + i];
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
}

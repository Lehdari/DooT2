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
static constexpr double weightDecay         = 1.0e-7; // TODO
static constexpr int    nTrainingEpochs     = 256;


ModelProto::ModelProto() :
    _optimizer  (_autoEncoder->parameters(), torch::optim::AdamOptions(2e-4).betas({0.9, 0.999}))
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

    cv::Mat imageOut(480, 640, CV_32FC4); // TODO temp

    // Move model parameters to GPU
    _autoEncoder->to(device);

    // Put data into a tensor
    std::vector<float> batchData(batchSize*4*480*640);
    Image<float> frame;
    for (int64_t epoch=0; epoch<256; ++epoch) {
        for (int b=0; b<batchSize; ++b) {
            convertImage(storage[0/*TODO*/][b].bgraFrame, frame); // do inline conversion for now (TODO)
            // transpose the image and add it to the batch data vector
            for (int c=0; c<4; ++c) {
                for (int j=0; j<480; ++j) {
                    for (int i=0; i<640; ++i) {
                        batchData[b*4*480*640 + c*480*640 + j*640 + i] = frame.data()[j*640*4 + i*4 + c];
                    }
                }
            }
        }
        auto options = torch::TensorOptions().device(torch::kCPU).dtype<float>();
        torch::Tensor batchCPU = torch::from_blob(batchData.data(), {batchSize, 4, 480, 640}, options);
        //batchCPU = batchCPU.transpose(1,3); // BHWC to BCHW
        torch::Tensor batchGPU = batchCPU.to(device);

        _autoEncoder->zero_grad();
        torch::Tensor output = _autoEncoder->forward(batchGPU);
        torch::Tensor loss = torch::mse_loss(batchGPU, output);
        loss.backward();

        _optimizer.step();

        torch::Tensor outputCPU = output.to(torch::kCPU);
        auto* dataOut = outputCPU.data_ptr<float>();
        for (int c=0; c<4; ++c) {
            for (int j=0; j<480; ++j) {
                for (int i=0; i<640; ++i) {
                    imageOut.ptr<float>(j)[i*4 + c] = dataOut[0*4*480*640 + c*480*640 + j*640 + i];
                }
            }
        }
        cv::imshow("pred", imageOut);
        cv::waitKey(1);

        printf(
            "\r[%2ld/%2ld][%3ld/%3ld] loss: %.4f",
            epoch,
            256,//kNumberOfEpochs,
            0,//++batch_index,
            0,//batches_per_epoch,
            loss.item<float>());
        fflush(stdout);
    }
}

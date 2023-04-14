//
// Project: DooT2
// File: MultiLevelFrameEncoder.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class MultiLevelFrameEncoderImpl : public torch::nn::Module {
public:
    MultiLevelFrameEncoderImpl();

    torch::Tensor forward(
        torch::Tensor x5,
        torch::Tensor x4,
        torch::Tensor x3,
        torch::Tensor x2,
        torch::Tensor x1,
        torch::Tensor x0,
        double lossLevel
    );

private:
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn1;
    torch::nn::Conv2d           _conv1b;
    torch::nn::BatchNorm2d      _bn1b;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bn2;
    torch::nn::Conv2d           _conv2b;
    torch::nn::BatchNorm2d      _bn2b;
    torch::nn::Conv2d           _conv3;
    torch::nn::BatchNorm2d      _bn3;
    torch::nn::Conv2d           _conv3b;
    torch::nn::BatchNorm2d      _bn3b;
    torch::nn::Conv2d           _conv4;
    torch::nn::BatchNorm2d      _bn4;
    torch::nn::Conv2d           _conv4b;
    torch::nn::BatchNorm2d      _bn4b;
    torch::nn::Conv2d           _conv5;
    torch::nn::BatchNorm2d      _bn5;
    torch::nn::Conv2d           _conv5b;
    torch::nn::BatchNorm2d      _bn5b;
    torch::nn::Conv2d           _conv6;
    torch::nn::BatchNorm2d      _bn6;
    torch::nn::Conv2d           _conv7;
    torch::nn::BatchNorm2d      _bn7;
    torch::nn::Conv2d           _conv8;
};
TORCH_MODULE(MultiLevelFrameEncoder);

} // namespace ml

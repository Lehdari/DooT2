//
// Project: DooT2
// File: FrameEncoder.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


class FrameEncoderImpl : public torch::nn::Module {
public:
    FrameEncoderImpl();

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bnEnc1;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bnEnc2;
    torch::nn::Conv2d           _conv3;
    torch::nn::BatchNorm2d      _bnEnc3;
    torch::nn::Conv2d           _conv4;
    torch::nn::BatchNorm2d      _bnEnc4;
    torch::nn::Conv2d           _conv5;
    torch::nn::BatchNorm2d      _bnEnc5;
    torch::nn::Conv2d           _conv6;
    torch::nn::BatchNorm2d      _bnEnc6;
    torch::nn::Conv2d           _conv7;
    torch::nn::BatchNorm2d      _bnEnc7;
    torch::nn::Conv2d           _conv8;
    torch::nn::BatchNorm2d      _bnEnc8;
};
TORCH_MODULE(FrameEncoder);

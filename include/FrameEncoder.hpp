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

#include "ResNeXtModule.hpp"


class FrameEncoderImpl : public torch::nn::Module {
public:
    FrameEncoderImpl();

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    forward(torch::Tensor x);

private:
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn1;
    ResNeXtModule               _resNext1;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bn2;
    ResNeXtModule               _resNext2;
    torch::nn::Conv2d           _conv3;
    torch::nn::BatchNorm2d      _bn3;
    ResNeXtModule               _resNext3;
    torch::nn::Conv2d           _conv4;
    torch::nn::BatchNorm2d      _bn4;
    ResNeXtModule               _resNext4;
    torch::nn::Conv2d           _conv5;
    torch::nn::BatchNorm2d      _bn5;
    ResNeXtModule               _resNext5;
    torch::nn::Conv2d           _conv6;
    torch::nn::BatchNorm2d      _bn6;
    ResNeXtModule               _resNext6;
    torch::nn::Conv2d           _conv7;
    torch::nn::BatchNorm2d      _bn7;
    ResNeXtModule               _resNext7;
    torch::nn::Conv2d           _conv8;
};
TORCH_MODULE(FrameEncoder);

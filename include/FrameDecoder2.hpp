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


class FrameDecoder2Impl : public torch::nn::Module {
public:
    FrameDecoder2Impl();

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        torch::Tensor x0,
        torch::Tensor x1,
        torch::Tensor x2,
        torch::Tensor x3,
        torch::Tensor x4,
        torch::Tensor x5,
        torch::Tensor x6,
        double skipLevel = 0.0
    );

private:
    ResNeXtModule               _resNext1a;
    ResNeXtModule               _resNext1b;
    torch::nn::ConvTranspose2d  _convTranspose1;
    torch::nn::BatchNorm2d      _bn1;
    ResNeXtModule               _resNext2a;
    ResNeXtModule               _resNext2b;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::BatchNorm2d      _bn2;
    ResNeXtModule               _resNext3a;
    ResNeXtModule               _resNext3b;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bn3;
    torch::nn::Dropout          _dropout3;
    ResNeXtModule               _resNext4a;
    ResNeXtModule               _resNext4b;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bn4;
    torch::nn::Dropout          _dropout4;
    ResNeXtModule               _resNext5a;
    ResNeXtModule               _resNext5b;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::BatchNorm2d      _bn5;
    torch::nn::Dropout          _dropout5;
    ResNeXtModule               _resNext6a;
    ResNeXtModule               _resNext6b;
    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::BatchNorm2d      _bn6;
    torch::nn::Dropout          _dropout6;
    ResNeXtModule               _resNext7a;
    ResNeXtModule               _resNext7b;
    torch::nn::ConvTranspose2d  _convTranspose7;
    torch::nn::BatchNorm2d      _bn7;
    torch::nn::Conv2d           _conv8;
    torch::nn::BatchNorm2d      _bn8;
    torch::nn::Conv2d           _conv8_Y;
    torch::nn::Conv2d           _conv8_UV;
};
TORCH_MODULE(FrameDecoder2);

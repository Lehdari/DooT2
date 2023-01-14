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


class FrameDecoderImpl : public torch::nn::Module {
public:
    FrameDecoderImpl();

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d           _conv1;
    ResNeXtModule               _resNext1;
    torch::nn::BatchNorm2d      _bnDec1;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::BatchNorm2d      _bnDec2;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bnDec3;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bnDec4;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::BatchNorm2d      _bnDec5;
    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::BatchNorm2d      _bnDec6;
    torch::nn::ConvTranspose2d  _convTranspose7;
    torch::nn::BatchNorm2d      _bnDec7;
    ResNeXtModule               _resNext8;
    torch::nn::ConvTranspose2d  _convTranspose8;
};
TORCH_MODULE(FrameDecoder);

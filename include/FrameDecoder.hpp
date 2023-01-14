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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    ResNeXtModule               _resNext1;
    torch::nn::ConvTranspose2d  _convTranspose1;
    torch::nn::BatchNorm2d      _bnDec1;
    ResNeXtModule               _resNext2;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::BatchNorm2d      _bnDec2;
    ResNeXtModule               _resNext3;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bnDec3;
    ResNeXtModule               _resNext4;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bnDec4;
    ResNeXtModule               _resNext5;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::ConvTranspose2d  _convTranspose5b;
    torch::nn::BatchNorm2d      _bnDec5;
    ResNeXtModule               _resNext6;
    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::ConvTranspose2d  _convTranspose6b;
    torch::nn::BatchNorm2d      _bnDec6;
    ResNeXtModule               _resNext7;
    torch::nn::ConvTranspose2d  _convTranspose7;
};
TORCH_MODULE(FrameDecoder);

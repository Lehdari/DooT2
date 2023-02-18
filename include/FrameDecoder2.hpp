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
#include "FourierFeatureModule.hpp"


class FrameDecoder2Impl : public torch::nn::Module {
public:
    FrameDecoder2Impl();

    torch::Tensor forward(torch::Tensor x);

private:
    ResNeXtModule               _resNext1;
    torch::nn::ConvTranspose2d  _convTranspose1;
    torch::nn::BatchNorm2d      _bn1;
    ResNeXtModule               _resNext2;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::BatchNorm2d      _bn2;
    FourierFeatureModule        _fourier2;
    ResNeXtModule               _resNext3;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bn3;
    FourierFeatureModule        _fourier3;
    ResNeXtModule               _resNext4;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bn4;
    FourierFeatureModule        _fourier4;
    ResNeXtModule               _resNext5;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::BatchNorm2d      _bn5;
    FourierFeatureModule        _fourier5;
    ResNeXtModule               _resNext6;
    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::BatchNorm2d      _bn6;
    FourierFeatureModule        _fourier6;
    ResNeXtModule               _resNext7;
    torch::nn::ConvTranspose2d  _convTranspose7;
    torch::nn::BatchNorm2d      _bn7;
    FourierFeatureModule        _fourier7;
    torch::nn::Conv2d           _conv8;
    torch::nn::BatchNorm2d      _bn8;
    torch::nn::Conv2d           _conv8_Y;
    torch::nn::Conv2d           _conv8_UV;
};
TORCH_MODULE(FrameDecoder2);

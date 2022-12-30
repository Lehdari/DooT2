//
// Project: DooT2
// File: AutoEncoder.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


class AutoEncoderImpl : public torch::nn::Module {
public:
    AutoEncoderImpl();

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

    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::BatchNorm2d      _bnDec6;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::BatchNorm2d      _bnDec5;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bnDec4;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bnDec3;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::ConvTranspose2d  _convTranspose1;
};
TORCH_MODULE(AutoEncoder);

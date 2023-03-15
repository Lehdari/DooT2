//
// Project: DooT2
// File: FlowDecoder.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class FlowDecoderImpl : public torch::nn::Module {
public:
    FlowDecoderImpl();

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d           _conv1;
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
};
TORCH_MODULE(FlowDecoder);

} // namespace ml

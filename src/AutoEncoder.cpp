//
// Project: DooT2
// File: AutoEncoder.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "AutoEncoder.hpp"


using namespace torch;


AutoEncoderImpl::AutoEncoderImpl() :
    _conv           (nn::Conv2dOptions(4, 8, {4, 4}).stride({2, 2})),
    _convTranspose  (nn::ConvTranspose2dOptions(8, 4, {4, 4}).stride({2, 2}))
{
    register_module("conv", _conv);
    register_module("convTranspose", _convTranspose);
}

torch::Tensor AutoEncoderImpl::forward(torch::Tensor x)
{
    // Encoder
    x = torch::tanh(_conv(x));

    // Decoder
    x = torch::tanh(_convTranspose(x)) * 0.5f + 0.5f;

    return x;
}

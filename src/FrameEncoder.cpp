//
// Project: DooT2
// File: FrameEncoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FrameEncoder.hpp"


using namespace torch;


FrameEncoderImpl::FrameEncoderImpl() :
    _conv1          (nn::Conv2dOptions(4, 16, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bnEnc1         (nn::BatchNorm2dOptions(16)),
    _conv2          (nn::Conv2dOptions(16, 32, {5, 6}).stride({3, 4}).bias(false).padding(1)),
    _bnEnc2         (nn::BatchNorm2dOptions(32)),
    _conv3          (nn::Conv2dOptions(32, 64, {4, 4}).stride({2, 2}).bias(false).padding({1, 1})),
    _bnEnc3         (nn::BatchNorm2dOptions(64)),
    _conv4          (nn::Conv2dOptions(64, 128, {4, 4}).stride({2, 2}).bias(false).padding({1, 1})),
    _bnEnc4         (nn::BatchNorm2dOptions(128)),
    _conv5          (nn::Conv2dOptions(128, 256, {4, 4}).stride({2, 2}).bias(false).padding({1, 1})),
    _bnEnc5         (nn::BatchNorm2dOptions(256)),
    _conv6          (nn::Conv2dOptions(256, 512, {4, 4}).stride({2, 2}).bias(false).padding({1, 1})),
    _bnEnc6         (nn::BatchNorm2dOptions(512)),
    _conv7          (nn::Conv2dOptions(512, 512, {2, 2}).bias(false)),
    _bnEnc7         (nn::BatchNorm2dOptions(512)),
    _conv8          (nn::Conv2dOptions(512, 128, {1, 1}))
{
    register_module("conv1", _conv1);
    register_module("bnEnc1", _bnEnc1);
    register_module("conv2", _conv2);
    register_module("bnEnc2", _bnEnc2);
    register_module("conv3", _conv3);
    register_module("bnEnc3", _bnEnc3);
    register_module("conv4", _conv4);
    register_module("bnEnc4", _bnEnc4);
    register_module("conv5", _conv5);
    register_module("bnEnc5", _bnEnc5);
    register_module("conv6", _conv6);
    register_module("bnEnc6", _bnEnc6);
    register_module("conv7", _conv7);
    register_module("bnEnc7", _bnEnc7);
    register_module("conv8", _conv8);
}

torch::Tensor FrameEncoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    // Encoder
    x = torch::leaky_relu(_bnEnc1(_conv1(x)), leakyReluNegativeSlope); // 320x240x16
    x = torch::leaky_relu(_bnEnc2(_conv2(x)), leakyReluNegativeSlope); // 80x80x32
    x = torch::leaky_relu(_bnEnc3(_conv3(x)), leakyReluNegativeSlope); // 40x40x64
    x = torch::leaky_relu(_bnEnc4(_conv4(x)), leakyReluNegativeSlope); // 20x20x128
    x = torch::leaky_relu(_bnEnc5(_conv5(x)), leakyReluNegativeSlope); // 10x10x256
    x = torch::leaky_relu(_bnEnc6(_conv6(x)), leakyReluNegativeSlope); // 5x5x512
    x = torch::leaky_relu(_bnEnc7(_conv7(x)), leakyReluNegativeSlope); // 4x4x512
    x = _conv8(x); // 4x4x128
    x = torch::reshape(x, {-1, 2048});

    return x;
}

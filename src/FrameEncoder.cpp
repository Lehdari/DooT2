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
    _conv1          (nn::Conv2dOptions(3, 32, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn1            (nn::BatchNorm2dOptions(32)),
    _resNext1       (32, 32, 8, 8),
    _conv2          (nn::Conv2dOptions(32, 64, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn2            (nn::BatchNorm2dOptions(64)),
    _resNext2       (64, 64, 8, 16),
    _conv3          (nn::Conv2dOptions(64, 128, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn3            (nn::BatchNorm2dOptions(128)),
    _resNext3       (128, 128, 8, 32),
    _conv4          (nn::Conv2dOptions(128, 256, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn4            (nn::BatchNorm2dOptions(256)),
    _resNext4       (256, 256, 8, 64),
    _conv5          (nn::Conv2dOptions(256, 256, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn5            (nn::BatchNorm2dOptions(256)),
    _resNext5       (256, 256, 8, 64),
    _conv6          (nn::Conv2dOptions(256, 512, {5, 6}).stride({3, 4}).bias(false).padding(1)),
    _bn6            (nn::BatchNorm2dOptions(512)),
    _resNext6       (512, 512, 8, 128),
    _conv7          (nn::Conv2dOptions(512, 512, {2, 2}).bias(false)),
    _bn7            (nn::BatchNorm2dOptions(512)),
    _resNext7       (512, 256, 8, 64),
    _conv8          (nn::Conv2dOptions(256, 128, {1, 1}))
{
    register_module("conv1", _conv1);
    register_module("bn1", _bn1);
    register_module("resNext1", _resNext1);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    register_module("resNext2", _resNext2);
    register_module("conv3", _conv3);
    register_module("bn3", _bn3);
    register_module("resNext3", _resNext3);
    register_module("conv4", _conv4);
    register_module("bn4", _bn4);
    register_module("resNext4", _resNext4);
    register_module("conv5", _conv5);
    register_module("bn5", _bn5);
    register_module("resNext5", _resNext5);
    register_module("conv6", _conv6);
    register_module("bn6", _bn6);
    register_module("resNext6", _resNext6);
    register_module("conv7", _conv7);
    register_module("bn7", _bn7);
    register_module("resNext7", _resNext7);
    register_module("conv8", _conv8);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
    FrameEncoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    // Encoder
    x = torch::tanh(_bn1(_conv1(x))); // 320x240x32
    auto x6 = _resNext1(x);
    x = torch::tanh(_bn2(_conv2(x6))); // 160x120x64
    auto x5 = _resNext2(x);
    x = torch::tanh(_bn3(_conv3(x5))); // 80x60x128
    auto x4 = _resNext3(x);
    x = torch::tanh(_bn4(_conv4(x4))); // 40x30x256
    auto x3 = _resNext4(x);
    x = torch::tanh(_bn5(_conv5(x3))); // 20x15x256
    auto x2 = _resNext5(x);
    x = torch::tanh(_bn6(_conv6(x2))); // 5x5x512
    auto x1 = _resNext6(x);
    x = torch::tanh(_bn7(_conv7(x1))); // 4x4x512
    auto x0 = _resNext7(x); // 4x4x256
    x = _conv8(x0); // 4x4x128
    x = torch::reshape(x, {-1, 2048});

    return {x, x0, x1, x2, x3, x4, x5, x6};
}

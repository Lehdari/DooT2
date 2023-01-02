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
    _conv8          (nn::Conv2dOptions(512, 128, {1, 1}).bias(false)),
    _bnEnc8         (nn::BatchNorm2dOptions(128)),

    _convTranspose8 (nn::ConvTranspose2dOptions(128, 512, {1, 1})),
    _bnDec8         (nn::BatchNorm2dOptions(512)),
    _convTranspose7 (nn::ConvTranspose2dOptions(512, 512, {2, 2})),
    _bnDec7         (nn::BatchNorm2dOptions(512)),
    _convTranspose6 (nn::ConvTranspose2dOptions(512, 256, {4, 4}).stride({2, 2})),
    _bnDec6         (nn::BatchNorm2dOptions(256)),
    _convTranspose5 (nn::ConvTranspose2dOptions(256, 128, {4, 4}).stride({2, 2})),
    _bnDec5         (nn::BatchNorm2dOptions(128)),
    _convTranspose4 (nn::ConvTranspose2dOptions(128, 64, {4, 4}).stride({2, 2})),
    _bnDec4         (nn::BatchNorm2dOptions(64)),
    _convTranspose3 (nn::ConvTranspose2dOptions(64, 32, {4, 4}).stride({2, 2})),
    _bnDec3         (nn::BatchNorm2dOptions(32)),
    _convTranspose2 (nn::ConvTranspose2dOptions(32, 16, {5, 6}).stride({3, 4})),
    _convTranspose1 (nn::ConvTranspose2dOptions(16, 4, {4, 4}).stride({2, 2}))
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
    register_module("bnEnc8", _bnEnc8);
    register_module("convTranspose8", _convTranspose8);
    register_module("bnDec8", _bnDec8);
    register_module("convTranspose7", _convTranspose7);
    register_module("bnDec7", _bnDec7);
    register_module("convTranspose6", _convTranspose6);
    register_module("bnDec6", _bnDec6);
    register_module("convTranspose5", _convTranspose5);
    register_module("bnDec5", _bnDec5);
    register_module("convTranspose4", _convTranspose4);
    register_module("bnDec4", _bnDec4);
    register_module("convTranspose3", _convTranspose3);
    register_module("bnDec3", _bnDec3);
    register_module("convTranspose2", _convTranspose2);
    register_module("convTranspose1", _convTranspose1);
}

torch::Tensor AutoEncoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    // Encoder
    x = torch::tanh(_bnEnc1(_conv1(x))); // 320x240x8
    x = torch::tanh(_bnEnc2(_conv2(x))); // 80x80x32
    x = torch::tanh(_bnEnc3(_conv3(x))); // 40x40x64
    x = torch::tanh(_bnEnc4(_conv4(x))); // 20x20x128
    x = torch::tanh(_bnEnc5(_conv5(x))); // 10x10x256
    x = torch::tanh(_bnEnc6(_conv6(x))); // 5x5x512
    x = torch::tanh(_bnEnc7(_conv7(x))); // 4x4x512
    x = torch::tanh(_bnEnc8(_conv8(x))); // 4x4x128

    // Decoder
    x = torch::tanh(_bnDec8(_convTranspose8(x)));
    x = torch::tanh(_bnDec7(_convTranspose7(x)));
    x = torch::tanh(_bnDec6(_convTranspose6(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec5(_convTranspose5(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec4(_convTranspose4(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec3(_convTranspose3(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_convTranspose2(x));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_convTranspose1(x)) * 0.5f + 0.5f;
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});

    return x;
}

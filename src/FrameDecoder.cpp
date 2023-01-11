//
// Project: DooT2
// File: FrameDecoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FrameDecoder.hpp"


using namespace torch;


FrameDecoderImpl::FrameDecoderImpl() :
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

torch::Tensor FrameDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    // Decoder
    x = torch::reshape(x, {-1, 128, 4, 4});
    x = torch::tanh(_bnDec8(_convTranspose8(x))); // 4x4x512
    x = torch::tanh(_bnDec7(_convTranspose7(x))); // 5x5x512
    x = torch::tanh(_bnDec6(_convTranspose6(x))); // 10x10x256
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec5(_convTranspose5(x))); // 20x20x128
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec4(_convTranspose4(x))); // 40x40x64
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_bnDec3(_convTranspose3(x))); // 80x80x32
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_convTranspose2(x)); // 320x240x16
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    x = torch::tanh(_convTranspose1(x)) * 0.5f + 0.5f; // 640x480x4
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});

    return x;
}

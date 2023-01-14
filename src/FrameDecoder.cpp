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
    _conv1          (nn::Conv2dOptions(128, 512, {3, 3}).bias(false).padding(1)),
    _bnDec1         (nn::BatchNorm2dOptions(512)),
    _resNext1       (512, 64, 8, 512),
    _convTranspose2 (nn::ConvTranspose2dOptions(512, 512, {2, 2}).bias(false)),
    _bnDec2         (nn::BatchNorm2dOptions(512)),
    _convTranspose3 (nn::ConvTranspose2dOptions(512, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec3         (nn::BatchNorm2dOptions(256)),
    _convTranspose4 (nn::ConvTranspose2dOptions(256, 128, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec4         (nn::BatchNorm2dOptions(128)),
    _convTranspose5 (nn::ConvTranspose2dOptions(128, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec5         (nn::BatchNorm2dOptions(64)),
    _convTranspose6 (nn::ConvTranspose2dOptions(64, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec6         (nn::BatchNorm2dOptions(64)),
    _convTranspose7 (nn::ConvTranspose2dOptions(64, 32, {5, 6}).stride({3, 4})),
    _bnDec7         (nn::BatchNorm2dOptions(32)),
    _resNext8       (32, 8, 4, 32),
    _convTranspose8 (nn::ConvTranspose2dOptions(32, 4, {4, 4}).stride({2, 2}))
{
    register_module("conv1", _conv1);
    register_module("bnDec1", _bnDec1);
    register_module("resNext1", _resNext1);
    register_module("convTranspose2", _convTranspose2);
    register_module("bnDec2", _bnDec2);
    register_module("convTranspose3", _convTranspose3);
    register_module("bnDec3", _bnDec3);
    register_module("convTranspose4", _convTranspose4);
    register_module("bnDec4", _bnDec4);
    register_module("convTranspose5", _convTranspose5);
    register_module("bnDec5", _bnDec5);
    register_module("convTranspose6", _convTranspose6);
    register_module("bnDec6", _bnDec6);
    register_module("convTranspose7", _convTranspose7);
    register_module("bnDec7", _bnDec7);
    register_module("resNext8", _resNext8);
    register_module("convTranspose8", _convTranspose8);
}

torch::Tensor FrameDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    // Decoder
    x = torch::reshape(x, {-1, 128, 4, 4});
    //x = x + 0.1f*torch::randn(x.sizes(), TensorOptions().device(x.device()));
    x = torch::tanh(_bnDec1(_conv1(x))); // 4x4x512
    x = _resNext1(x);
    //x = x + 0.05f*torch::randn(x.sizes(), TensorOptions().device(x.device()));
    x = torch::tanh(_bnDec2(_convTranspose2(x))); // 5x5x512
    //x = x + 0.025f*torch::randn(x.sizes(), TensorOptions().device(x.device()));
    x = torch::tanh(_bnDec3(_convTranspose3(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 10x10x256
    x = torch::tanh(_bnDec4(_convTranspose4(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x20x128
    x = torch::tanh(_bnDec5(_convTranspose5(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x40x64
    x = torch::tanh(_bnDec6(_convTranspose6(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x80x32
    x = torch::tanh(_bnDec7(_convTranspose7(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x16
    x = _resNext8(x);
    x = torch::tanh(_convTranspose8(x)) * 0.5f + 0.5f;
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 640x480x4

    return x;
}

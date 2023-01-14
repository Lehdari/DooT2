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
    _resNext1       (128, 32, 8, 256),
    _convTranspose1 (nn::ConvTranspose2dOptions(256, 256, {2, 2}).bias(false)),
    _bnDec1         (nn::BatchNorm2dOptions(256)),
    _resNext2       (256, 64, 8, 512),
    _convTranspose2 (nn::ConvTranspose2dOptions(512, 256, {5, 6}).stride({3, 4}).bias(false)),
    _bnDec2         (nn::BatchNorm2dOptions(256)),
    _resNext3       (256, 64, 8, 512),
    _convTranspose3 (nn::ConvTranspose2dOptions(512, 512, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec3         (nn::BatchNorm2dOptions(512)),
    _resNext4       (512, 32, 8, 256),
    _convTranspose4 (nn::ConvTranspose2dOptions(256, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec4         (nn::BatchNorm2dOptions(256)),
    _resNext5       (256, 16, 8, 128),
    _convTranspose5 (nn::ConvTranspose2dOptions(128, 128, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec5         (nn::BatchNorm2dOptions(128)),
    _resNext6       (128, 8, 8, 64),
    _convTranspose6 (nn::ConvTranspose2dOptions(64, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec6         (nn::BatchNorm2dOptions(64)),
    _resNext7       (64, 8, 4, 32),
    _convTranspose7 (nn::ConvTranspose2dOptions(32, 4, {4, 4}).stride({2, 2}))
{
    register_module("resNext1", _resNext1);
    register_module("convTranspose1", _convTranspose1);
    register_module("bnDec1", _bnDec1);
    register_module("resNext2", _resNext2);
    register_module("convTranspose2", _convTranspose2);
    register_module("bnDec2", _bnDec2);
    register_module("resNext3", _resNext3);
    register_module("convTranspose3", _convTranspose3);
    register_module("bnDec3", _bnDec3);
    register_module("resNext4", _resNext4);
    register_module("convTranspose4", _convTranspose4);
    register_module("bnDec4", _bnDec4);
    register_module("resNext5", _resNext5);
    register_module("convTranspose5", _convTranspose5);
    register_module("bnDec5", _bnDec5);
    register_module("resNext6", _resNext6);
    register_module("convTranspose6", _convTranspose6);
    register_module("bnDec6", _bnDec6);
    register_module("resNext7", _resNext7);
    register_module("convTranspose7", _convTranspose7);
}

torch::Tensor FrameDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    // Decoder
    x = torch::reshape(x, {-1, 128, 4, 4});
    x = torch::tanh(_bnDec1(_convTranspose1(_resNext1(x)))); // 5x5x256
    x = torch::tanh(_bnDec2(_convTranspose2(_resNext2(x))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x15x256
    x = torch::tanh(_bnDec3(_convTranspose3(_resNext3(x))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x30x512
    x = torch::tanh(_bnDec4(_convTranspose4(_resNext4(x))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x60x256
    x = torch::tanh(_bnDec5(_convTranspose5(_resNext5(x))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 160x120x128
    x = torch::tanh(_bnDec6(_convTranspose6(_resNext6(x))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x64
    x = torch::tanh(_convTranspose7(_resNext7(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 640x480x4

    return x;
}

//
// Project: DooT2
// File: FrameDecoder2.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FrameDecoder2.hpp"


using namespace torch;


FrameDecoder2Impl::FrameDecoder2Impl() :
    _resNext1           (128, 32, 8, 256),
    _convTranspose1     (nn::ConvTranspose2dOptions(256, 256, {2, 2}).bias(false)),
    _bn1                (nn::BatchNorm2dOptions(256)),
    _resNext2           (256, 64, 8, 512),
    _convTranspose2     (nn::ConvTranspose2dOptions(512, 256, {5, 6}).stride({3, 4}).bias(false)),
    _bn2                (nn::BatchNorm2dOptions(256)),
    _fourier2           (256, 64, 1.0, 2.0),
    _resNext3           (384, 64, 8, 512),
    _convTranspose3     (nn::ConvTranspose2dOptions(512, 512, {4, 4}).stride({2, 2}).bias(false)),
    _bn3                (nn::BatchNorm2dOptions(512)),
    _fourier3           (512, 128, 1.0, 4.0),
    _resNext4           (768, 32, 8, 256),
    _convTranspose4     (nn::ConvTranspose2dOptions(256, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bn4                (nn::BatchNorm2dOptions(256)),
    _fourier4           (256, 128, 2.0, 8.0),
    _resNext5           (512, 16, 8, 128),
    _convTranspose5     (nn::ConvTranspose2dOptions(128, 128, {4, 4}).stride({2, 2}).bias(false)),
    _bn5                (nn::BatchNorm2dOptions(128)),
    _fourier5           (128, 64, 4.0, 16.0),
    _resNext6           (256, 8, 8, 64),
    _convTranspose6     (nn::ConvTranspose2dOptions(64, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bn6                (nn::BatchNorm2dOptions(64)),
    _fourier6           (64, 32, 8.0, 32.0),
    _resNext7           (128, 8, 4, 32),
    _convTranspose7     (nn::ConvTranspose2dOptions(32, 32, {4, 4}).stride({2, 2}).bias(false)),
    _bn7                (nn::BatchNorm2dOptions(32)),
    _fourier7           (32, 16, 16.0, 64.0),
    _conv8              (nn::Conv2dOptions(64, 32, {1, 1}).bias(false)),
    _bn8                (nn::BatchNorm2dOptions(32)),
    _conv8_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv8_UV           (nn::Conv2dOptions(16, 2, {1, 1}))
{
    register_module("resNext1", _resNext1);
    register_module("convTranspose1", _convTranspose1);
    register_module("bn1", _bn1);
    register_module("resNext2", _resNext2);
    register_module("convTranspose2", _convTranspose2);
    register_module("bn2", _bn2);
    register_module("fourier2", _fourier2);
    register_module("resNext3", _resNext3);
    register_module("convTranspose3", _convTranspose3);
    register_module("bn3", _bn3);
    register_module("fourier3", _fourier3);
    register_module("resNext4", _resNext4);
    register_module("convTranspose4", _convTranspose4);
    register_module("bn4", _bn4);
    register_module("fourier4", _fourier4);
    register_module("resNext5", _resNext5);
    register_module("convTranspose5", _convTranspose5);
    register_module("bn5", _bn5);
    register_module("fourier5", _fourier5);
    register_module("resNext6", _resNext6);
    register_module("convTranspose6", _convTranspose6);
    register_module("bn6", _bn6);
    register_module("fourier6", _fourier6);
    register_module("resNext7", _resNext7);
    register_module("convTranspose7", _convTranspose7);
    register_module("bn7", _bn7);
    register_module("fourier7", _fourier7);
    register_module("conv8", _conv8);
    register_module("bn8", _bn8);
    register_module("conv8_Y", _conv8_Y);
    register_module("conv8_UV", _conv8_UV);
}

torch::Tensor FrameDecoder2Impl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    int batchSize = x.sizes()[0];

    // Decoder
    x = torch::reshape(x, {batchSize, 128, 4, 4});
    x = torch::leaky_relu(_bn1(_convTranspose1(_resNext1(x))), leakyReluNegativeSlope); // 5x5x256
    x = torch::leaky_relu(_bn2(_convTranspose2(_resNext2(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x15x256
    x = _fourier2(x);
    x = torch::leaky_relu(_bn3(_convTranspose3(_resNext3(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x30x512
    x = _fourier3(x);
    x = torch::leaky_relu(_bn4(_convTranspose4(_resNext4(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x60x256
    x = _fourier4(x);
    x = torch::leaky_relu(_bn5(_convTranspose5(_resNext5(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 160x120x128
    x = _fourier5(x);
    x = torch::leaky_relu(_bn6(_convTranspose6(_resNext6(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x64
    x = _fourier6(x);
    x = torch::leaky_relu(_bn7(_convTranspose7(_resNext7(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 640x480x32
    x = _fourier7(x);
    x = torch::leaky_relu(_bn8(_conv8(x)), leakyReluNegativeSlope);

    torch::Tensor s_Y = 0.5f + 0.51f * torch::tanh(_conv8_Y(
        x.index({Slice(), Slice(None, 16), Slice(), Slice()}))); // 640x480x1
    torch::Tensor s_UV = 0.51f * torch::tanh(_conv8_UV(
        x.index({Slice(), Slice(16, None), Slice(), Slice()}))); // 640x480x2
    torch::Tensor s = torch::cat({s_Y, s_UV}, 1);

    return s;
}

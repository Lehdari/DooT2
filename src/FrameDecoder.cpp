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
    _resNext1           (128, 32, 8, 256),
    _convTranspose1     (nn::ConvTranspose2dOptions(256, 256, {2, 2}).bias(false)),
    _bnDec1             (nn::BatchNorm2dOptions(256)),
    _resNext2           (256, 64, 8, 512),
    _convTranspose2     (nn::ConvTranspose2dOptions(512, 256, {5, 6}).stride({3, 4}).bias(false)),
    _bnDec2             (nn::BatchNorm2dOptions(256)),
    _resNext3           (256, 64, 8, 512),
    _convTranspose3     (nn::ConvTranspose2dOptions(512, 512, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec3             (nn::BatchNorm2dOptions(512)),
    _resNext4           (512, 32, 8, 256),
    _convTranspose4     (nn::ConvTranspose2dOptions(256, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bnDec4             (nn::BatchNorm2dOptions(256)),
    _resNext5           (256, 16, 8, 128),
    _convTranspose5     (nn::ConvTranspose2dOptions(128, 128, {4, 4}).stride({2, 2}).bias(false)),
    _convTranspose5_Y   (nn::ConvTranspose2dOptions(64, 1, {1, 1})),
    _convTranspose5_UV  (nn::ConvTranspose2dOptions(64, 2, {1, 1})),
    _bnDec5             (nn::BatchNorm2dOptions(128)),
    _resNext6           (128, 8, 8, 64),
    _convTranspose6     (nn::ConvTranspose2dOptions(64, 64, {4, 4}).stride({2, 2}).bias(false)),
    _convTranspose6_Y   (nn::ConvTranspose2dOptions(32, 1, {1, 1})),
    _convTranspose6_UV  (nn::ConvTranspose2dOptions(32, 2, {1, 1})),
    _bnDec6             (nn::BatchNorm2dOptions(64)),
    _resNext7           (64, 8, 4, 32),
    _convTranspose7_Y   (nn::ConvTranspose2dOptions(16, 1, {4, 4}).stride({2, 2})),
    _convTranspose7_UV  (nn::ConvTranspose2dOptions(16, 2, {4, 4}).stride({2, 2}))
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
    register_module("convTranspose5_Y", _convTranspose5_Y);
    register_module("convTranspose5_UV", _convTranspose5_UV);
    register_module("bnDec5", _bnDec5);
    register_module("resNext6", _resNext6);
    register_module("convTranspose6", _convTranspose6);
    register_module("convTranspose6_Y", _convTranspose6_Y);
    register_module("convTranspose6_UV", _convTranspose6_UV);
    register_module("bnDec6", _bnDec6);
    register_module("resNext7", _resNext7);
    register_module("convTranspose7_Y", _convTranspose7_Y);
    register_module("convTranspose7_UV", _convTranspose7_UV);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FrameDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    // Decoder
    x = torch::reshape(x, {-1, 128, 4, 4});
    x = torch::leaky_relu(_bnDec1(_convTranspose1(_resNext1(x))), leakyReluNegativeSlope); // 5x5x256
    x = torch::leaky_relu(_bnDec2(_convTranspose2(_resNext2(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x15x256
    x = torch::leaky_relu(_bnDec3(_convTranspose3(_resNext3(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x30x512
    x = torch::leaky_relu(_bnDec4(_convTranspose4(_resNext4(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x60x256
    x = torch::leaky_relu(_bnDec5(_convTranspose5(_resNext5(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 160x120x128

    // 4x downscaled output
    torch::Tensor s4_Y = 0.5f + 0.5f * torch::tanh(_convTranspose5_Y(
        x.index({Slice(), Slice(None, 64), Slice(), Slice()})));
    torch::Tensor s4_UV = 0.5f * torch::tanh(_convTranspose5_UV(
        x.index({Slice(), Slice(None, 64), Slice(), Slice()})));
    torch::Tensor s4 = torch::cat({s4_Y, s4_UV}, 1);

    x = torch::leaky_relu(_bnDec6(_convTranspose6(_resNext6(x))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x64

    // 2x downscaled output
    torch::Tensor s2_Y = 0.5f + 0.5f * torch::tanh(_convTranspose6_Y(
        x.index({Slice(), Slice(None, 32), Slice(), Slice()})));
    torch::Tensor s2_UV = 0.5f * torch::tanh(_convTranspose6_UV(
        x.index({Slice(), Slice(None, 32), Slice(), Slice()})));
    torch::Tensor s2 = torch::cat({s2_Y, s2_UV}, 1);

    x = _resNext7(x);
    torch::Tensor s_Y = 0.5f + 0.5f * torch::tanh(_convTranspose7_Y(
        x.index({Slice(), Slice(None, 16), Slice(), Slice()}))); // 640x480x1
    s_Y = s_Y.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    torch::Tensor s_UV = 0.5f * torch::tanh(_convTranspose7_UV(
        x.index({Slice(), Slice(16, None), Slice(), Slice()}))); // 640x480x2
    s_UV = s_UV.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});
    torch::Tensor s = torch::cat({s_Y, s_UV}, 1);

    return std::make_tuple(s, s2, s4);
}

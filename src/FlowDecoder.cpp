//
// Project: DooT2
// File: FlowDecoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FlowDecoder.hpp"


using namespace torch;


FlowDecoderImpl::FlowDecoderImpl() :
    _conv1          (nn::Conv2dOptions(128, 128, {3, 3}).bias(false).padding(1)),
    _bnDec1         (nn::BatchNorm2dOptions(128)),
    _convTranspose2 (nn::ConvTranspose2dOptions(128, 64, {2, 2}).bias(false)),
    _bnDec2         (nn::BatchNorm2dOptions(64)),
    _convTranspose3 (nn::ConvTranspose2dOptions(64, 32, {4, 4}).bias(false).stride({2, 2})),
    _bnDec3         (nn::BatchNorm2dOptions(32)),
    _convTranspose4 (nn::ConvTranspose2dOptions(32, 16, {4, 4}).bias(false).stride({2, 2})),
    _bnDec4         (nn::BatchNorm2dOptions(16)),
    _convTranspose5 (nn::ConvTranspose2dOptions(16, 8, {4, 4}).bias(false).stride({2, 2})),
    _bnDec5         (nn::BatchNorm2dOptions(8)),
    _convTranspose6 (nn::ConvTranspose2dOptions(8, 2, {4, 4}).stride({2, 2}))
{
    register_module("conv1", _conv1);
    register_module("bnDec1", _bnDec1);
    register_module("convTranspose2", _convTranspose2);
    register_module("bnDec2", _bnDec2);
    register_module("convTranspose3", _convTranspose3);
    register_module("bnDec3", _bnDec3);
    register_module("convTranspose4", _convTranspose4);
    register_module("bnDec4", _bnDec4);
    register_module("convTranspose5", _convTranspose5);
    register_module("bnDec5", _bnDec5);
    register_module("convTranspose6", _convTranspose6);

    auto w = _convTranspose5->named_parameters(false).find("weight");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _convTranspose6->named_parameters(false).find("weight");
    torch::nn::init::normal_(*w, 0.0, 0.001);
}

torch::Tensor FlowDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    // Decoder
    x = torch::reshape(x, {-1, 128, 4, 4});
    x = torch::leaky_relu(_bnDec1(_conv1(x)), leakyReluNegativeSlope); // 4x4x512
    x = torch::leaky_relu(_bnDec2(_convTranspose2(x)), leakyReluNegativeSlope); // 5x5x512
    x = torch::leaky_relu(_bnDec3(_convTranspose3(x)), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});  // 10x10x256
    x = torch::leaky_relu(_bnDec4(_convTranspose4(x)), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x20x128
    x = torch::tanh(_bnDec5(_convTranspose5(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x40x64
    x = _convTranspose6(x);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x80x32
    x = nn::functional::interpolate(x, nn::functional::InterpolateFuncOptions()
        .mode(kBilinear)
        .size(std::vector<int64_t>{480, 640})
        .align_corners(false));
    x = x.transpose(1, 3);
    x = x.transpose(1, 2);

    return x;
}

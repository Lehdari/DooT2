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
    _linear1        (nn::LinearOptions(2048, 2048)),
    _convTranspose1 (nn::ConvTranspose2dOptions(128, 128, {1, 1}).bias(false)),
    _bnDec1         (nn::BatchNorm2dOptions(128)),
    _convTranspose2 (nn::ConvTranspose2dOptions(128, 128, {2, 2}).bias(false)),
    _bnDec2         (nn::BatchNorm2dOptions(128)),
    _convTranspose3 (nn::ConvTranspose2dOptions(128, 64, {4, 4}).bias(false).stride({2, 2})),
    _bnDec3         (nn::BatchNorm2dOptions(64)),
    _convTranspose4 (nn::ConvTranspose2dOptions(64, 32, {4, 4}).bias(false).stride({2, 2})),
    _bnDec4         (nn::BatchNorm2dOptions(32)),
    _convTranspose5 (nn::ConvTranspose2dOptions(32, 16, {4, 4}).bias(false).stride({2, 2})),
    _bnDec5         (nn::BatchNorm2dOptions(16)),
    _convTranspose6 (nn::ConvTranspose2dOptions(16, 2, {4, 4}).stride({2, 2}))
{
    register_module("linear1", _linear1);
    register_module("convTranspose1", _convTranspose1);
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

    auto w = _linear1->named_parameters(false).find("weight");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _convTranspose5->named_parameters(false).find("weight");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _convTranspose6->named_parameters(false).find("weight");
    torch::nn::init::normal_(*w, 0.0, 0.001);
}

torch::Tensor FlowDecoderImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    // Decoder
    x = torch::reshape(x, {-1, 2048});
    x = x + torch::tanh(_linear1(x)); // residual dense layer
    x = torch::reshape(x, {-1, 128, 4, 4});
    x = torch::tanh(_bnDec1(_convTranspose1(x))); // 4x4x512
    x = torch::tanh(_bnDec2(_convTranspose2(x))); // 5x5x512
    x = torch::tanh(_bnDec3(_convTranspose3(x)));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});  // 10x10x256
    x = torch::tanh(_bnDec4(_convTranspose4(x)));
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

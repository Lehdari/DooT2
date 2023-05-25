//
// Project: DooT2
// File: MultiLevelDecoderModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelDecoderModule.hpp"


using namespace ml;
using namespace torch;


MultiLevelDecoderModuleImpl::MultiLevelDecoderModuleImpl(
    double level,
    int inputChannels,
    int hiddenChannels1,
    int hiddenChannels2,
    int outputChannels,
    int nGroups1,
    int nGroups2,
    int groupChannels1,
    int groupChannels2,
    int outputWidth,
    int outputHeight,
    const ExpandingArray<2>& kernelSize,
    const ExpandingArray<2>& stride,
    const torch::indexing::Slice& vSlice,
    const torch::indexing::Slice& hSlice
) :
    _level          (level),
    _outputWidth    (outputWidth),
    _outputHeight   (outputHeight),
    _vSlice         (vSlice),
    _hSlice         (hSlice),
    _resNext1       (inputChannels, hiddenChannels1, nGroups1, groupChannels1),
    _resNext2       (hiddenChannels1, hiddenChannels2, nGroups2, groupChannels2),
    _convTranspose  (nn::ConvTranspose2dOptions(hiddenChannels2, outputChannels, kernelSize)
                     .stride(stride).bias(false)),
    _bnMain         (nn::BatchNorm2dOptions(outputChannels)),
    _convAux        (nn::Conv2dOptions(outputChannels, 16, {3, 3}).bias(false).padding(1)),
    _bnAux          (nn::BatchNorm2dOptions(16)),
    _conv_Y         (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv_UV        (nn::Conv2dOptions(16, 2, {1, 1}))
{
    register_module("resNext1", _resNext1);
    register_module("resNext2", _resNext2);
    register_module("convTranspose", _convTranspose);
    register_module("bnMain", _bnMain);
    register_module("convAux", _convAux);
    register_module("bnAux", _bnAux);
    register_module("conv_Y", _conv_Y);
    register_module("conv_UV", _conv_UV);

    auto* w = _conv_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

std::tuple<torch::Tensor, torch::Tensor> MultiLevelDecoderModuleImpl::forward(torch::Tensor x, double level)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor y;
    if (level > _level) {
        x = torch::leaky_relu(_bnMain(_convTranspose(_resNext2(_resNext1(x)))), leakyReluNegativeSlope);
        x = x.index({Slice(), Slice(), _vSlice, _hSlice});

        // auxiliary image output
        y = torch::leaky_relu(_bnAux(_convAux(x)), leakyReluNegativeSlope);
        torch::Tensor y_Y = 0.5f + 0.51f * torch::tanh(_conv_Y(y));
        torch::Tensor y_UV = 0.51f * torch::tanh(_conv_UV(y));
        y = torch::cat({y_Y, y_UV}, 1);
    }
    else
        y = torch::zeros({x.sizes()[0], 3, _outputHeight, _outputWidth}, TensorOptions().device(x.device()));

    return {x, y};
}
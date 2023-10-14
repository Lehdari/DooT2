//
// Project: DooT2
// File: ResNetConvBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ResNetConvBlock.hpp"


using namespace ml;
using namespace torch;


ResNetConvBlockImpl::ResNetConvBlockImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int groups,
    bool useSqueezeExcitation,
    double normalInitializationStd,
    bool useEdgePadding
) :
    _skipLayer              (inputChannels != outputChannels),
    _useSqueezeExcitation   (useSqueezeExcitation),
    _useEdgePadding         (useEdgePadding),
    _conv1                  (nn::Conv2dOptions(inputChannels, hiddenChannels, {1, 1}).bias(false)),
    _bn1                    (nn::BatchNorm2dOptions(hiddenChannels)),
    _conv2                  (nn::Conv2dOptions(hiddenChannels, hiddenChannels, {3, 3}).bias(false).groups(groups)),
    _bn2                    (nn::BatchNorm2dOptions(hiddenChannels)),
    _se1                    (hiddenChannels, outputChannels, hiddenChannels),
    _conv3                  (nn::Conv2dOptions(hiddenChannels, outputChannels, {1, 1}).bias(false)),
    _convSkip               (nn::Conv2dOptions(inputChannels, outputChannels, {1, 1}).bias(false))
{
    register_module("conv1", _conv1);
    register_module("bn1", _bn1);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    if (_useSqueezeExcitation)
        register_module("se1", _se1);
    register_module("conv3", _conv3);
    if (_skipLayer)
        register_module("convSkip", _convSkip);

    if (normalInitializationStd > 0.0) {
        auto* w = _conv3->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
    }
}

torch::Tensor ResNetConvBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y = _conv1(x);
    y = gelu(_bn1(y), "tanh");
    y = _useEdgePadding ? torch::reflection_pad2d(y, {1,1,1,1}) : torch::pad(y, {1,1,1,1});
    y = _conv2(y);
    y = gelu(_bn2(y), "tanh");
    if (_useSqueezeExcitation)
        y = _se1(y);
    y = _conv3(y);

    if (_skipLayer)
        x = _convSkip(x);

    return x + y;
}

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
    double reluAlpha,
    double normalInitializationStd
) :
    _reluAlpha  (reluAlpha),
    _skipLayer  (inputChannels != outputChannels),
    _bn1        (nn::BatchNorm2dOptions(inputChannels)),
    _conv1      (nn::Conv2dOptions(inputChannels, hiddenChannels, {3, 3}).bias(false).padding(1)),
    _bn2        (nn::BatchNorm2dOptions(hiddenChannels)),
    _conv2      (nn::Conv2dOptions(hiddenChannels, outputChannels, {3, 3}).bias(false).padding(1)),
    _convSkip   (nn::Conv2dOptions(inputChannels, outputChannels, {1, 1}).bias(false))
{
    register_module("bn1", _bn1);
    register_module("conv1", _conv1);
    register_module("bn2", _bn2);
    register_module("conv2", _conv2);
    if (_skipLayer)
        register_module("convSkip", _convSkip);

    if (normalInitializationStd > 0.0) {
        auto* w = _conv2->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
    }
}

torch::Tensor ResNetConvBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y = _conv1(leaky_relu(_bn1(x), _reluAlpha));
    y = _conv2(leaky_relu(_bn2(y), _reluAlpha));

    if (_skipLayer)
        x = _convSkip(x);

    return x + y;
}

//
// Project: DooT2
// File: ResNetDownscaleConvBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ResNetDownscaleConvBlock.hpp"


using namespace ml;
using namespace torch;


ResNetDownscaleConvBlockImpl::ResNetDownscaleConvBlockImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int xDownScale,
    int yDownScale,
    int groups,
    bool useSqueezeExcitation,
    double normalInitializationStd
) :
    _skipLayer              (inputChannels != outputChannels),
    _useSqueezeExcitation   (useSqueezeExcitation),
    _conv1                  (nn::Conv2dOptions(inputChannels, hiddenChannels, {1, 1}).bias(false)),
    _bn1                    (nn::BatchNorm2dOptions(hiddenChannels)),
    _conv2                  (nn::Conv2dOptions(hiddenChannels, hiddenChannels, {yDownScale+2, xDownScale+2})
                             .stride({yDownScale, xDownScale}).bias(false).padding({1, 1}).groups(groups)),
    _bn2                    (nn::BatchNorm2dOptions(hiddenChannels)),
    _se1                    (hiddenChannels, outputChannels, hiddenChannels),
    _conv3                  (nn::Conv2dOptions(hiddenChannels, outputChannels, {1, 1}).bias(false)),
    _avgPool1               (nn::AvgPool2dOptions({yDownScale, xDownScale})),
    _convSkip               (nn::Conv2dOptions(inputChannels, outputChannels, {1, 1}).bias(false))
{
    register_module("conv1", _conv1);
    register_module("bn2", _bn1);
    register_module("conv2", _conv2);
    register_module("bn3", _bn2);
    if (_useSqueezeExcitation)
        register_module("se1", _se1);
    register_module("conv3", _conv3);
    register_module("avgPool1", _avgPool1);
    if (_skipLayer)
        register_module("convSkip", _convSkip);

    if (normalInitializationStd > 0.0) {
        auto* w = _conv3->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
    }
}

torch::Tensor ResNetDownscaleConvBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y = _conv1(x);
    y = gelu(_bn1(y), "tanh");
    y = _conv2(y);
    y = gelu(_bn2(y), "tanh");
    if (_useSqueezeExcitation)
        y = _se1(y);
    y = _conv3(y);

    // Skip connection
    x = _avgPool1(x);
    if (_skipLayer)
        x = _convSkip(x);

    return x + y;
}

//
// Project: DooT2
// File: ResNetFourierConvBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ResNetFourierConvBlock.hpp"


using namespace ml;
using namespace torch;


ResNetFourierConvBlockImpl::ResNetFourierConvBlockImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int groups,
    double normalInitializationStd
) :
    _skipLayer              (inputChannels != outputChannels),
    _fourierConv1           (inputChannels, hiddenChannels, 0.5, groups, normalInitializationStd),
    _fourierConv2           (hiddenChannels, outputChannels, 0.5, groups, normalInitializationStd),
    _convSkip               (nn::Conv2dOptions(inputChannels, outputChannels, {1, 1}).bias(false))
{
    register_module("fourierConv1", _fourierConv1);
    register_module("fourierConv2", _fourierConv2);
    if (_skipLayer)
        register_module("convSkip", _convSkip);
}

torch::Tensor ResNetFourierConvBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y = _fourierConv2(_fourierConv1(x));

    if (_skipLayer)
        x = _convSkip(x);

    return x + y;
}

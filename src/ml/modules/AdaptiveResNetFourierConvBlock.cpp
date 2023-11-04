//
// Project: DooT2
// File: AdaptiveResNetFourierConvBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/AdaptiveResNetFourierConvBlock.hpp"


using namespace ml;
using namespace torch;


AdaptiveResNetFourierConvBlockImpl::AdaptiveResNetFourierConvBlockImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int contextChannels,
    int groups,
    int filterBankSize,
    double normalInitializationStd
) :
    _skipLayer              (inputChannels != outputChannels),
    _fourierConv1           (inputChannels, hiddenChannels, contextChannels, 0.5, groups, filterBankSize,
                             normalInitializationStd),
    _fourierConv2           (hiddenChannels, outputChannels, contextChannels, 0.5, groups, filterBankSize,
                             normalInitializationStd),
    _convSkip               (nn::Conv2dOptions(inputChannels, outputChannels, {1, 1}).bias(false))
{
    register_module("fourierConv1", _fourierConv1);
    register_module("fourierConv2", _fourierConv2);
    if (_skipLayer)
        register_module("convSkip", _convSkip);
}

torch::Tensor AdaptiveResNetFourierConvBlockImpl::forward(torch::Tensor x, const torch::Tensor& context)
{
    torch::Tensor y = _fourierConv2(_fourierConv1(x, context), context);

    if (_skipLayer)
        x = _convSkip(x);

    return x + y;
}

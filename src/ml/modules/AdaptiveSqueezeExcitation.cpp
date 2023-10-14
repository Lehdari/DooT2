//
// Project: DooT2
// File: AdaptiveSqueezeExcitation.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/AdaptiveSqueezeExcitation.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


AdaptiveSqueezeExcitationImpl::AdaptiveSqueezeExcitationImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    int contextChannels
) :
    _avgPool1   (nn::AdaptiveAvgPool2dOptions(1)),
    _linear1a   (nn::LinearOptions(inputChannels, hiddenChannels)),
    _linear1b   (nn::LinearOptions(contextChannels, hiddenChannels)),
    _linear2    (nn::LinearOptions(hiddenChannels, outputChannels))
{
    register_module("avgPool1", _avgPool1);
    register_module("linear1a", _linear1a);
    register_module("linear1b", _linear1b);
    register_module("linear2", _linear2);
}

torch::Tensor AdaptiveSqueezeExcitationImpl::forward(const torch::Tensor& x, const torch::Tensor& context)
{
    auto b = x.sizes()[0];
    auto c = x.sizes()[1];
    torch::Tensor y = _avgPool1(x).view({b, c});
    y = sigmoid(_linear2(gelu(_linear1a(y) + _linear1b(context), "tanh"))).view({b, c, 1, 1});
    return x * y;
}

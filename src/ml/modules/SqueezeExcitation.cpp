//
// Project: DooT2
// File: SqueezeExcitation.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/SqueezeExcitation.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


SqueezeExcitationImpl::SqueezeExcitationImpl(int inputChannels, int hiddenChannels, int outputChannels) :
    _avgPool1   (nn::AdaptiveAvgPool2dOptions(1)),
    _linear1    (nn::LinearOptions(inputChannels, hiddenChannels)),
    _linear2    (nn::LinearOptions(hiddenChannels, outputChannels))
{
    register_module("avgPool1", _avgPool1);
    register_module("linear1", _linear1);
    register_module("linear2", _linear2);
}

torch::Tensor SqueezeExcitationImpl::forward(const torch::Tensor& x)
{
    auto b = x.sizes()[0];
    auto c = x.sizes()[1];
    torch::Tensor y = _avgPool1(x).view({b, c});
    y = sigmoid(_linear2(gelu(_linear1(y), "tanh"))).view({b, c, 1, 1});
    return x * y;
}

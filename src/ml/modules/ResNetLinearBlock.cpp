//
// Project: DooT2
// File: ResNetLinearBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ResNetLinearBlock.hpp"


using namespace ml;
using namespace torch;


ResNetLinearBlockImpl::ResNetLinearBlockImpl(
    int inputChannels,
    int hiddenChannels,
    int outputChannels,
    double reluAlpha,
    double normalInitializationStd
) :
    _reluAlpha  (reluAlpha),
    _skipLayer  (inputChannels != outputChannels),
    _bn1        (nn::BatchNorm1dOptions(inputChannels)),
    _linear1    (nn::LinearOptions(inputChannels, hiddenChannels).bias(false)),
    _bn2        (nn::BatchNorm1dOptions(hiddenChannels)),
    _linear2    (nn::LinearOptions(hiddenChannels, outputChannels).bias(false)),
    _linearSkip (nn::LinearOptions(inputChannels, outputChannels).bias(false))
{
    register_module("bn1", _bn1);
    register_module("linear1", _linear1);
    register_module("bn2", _bn2);
    register_module("linear2", _linear2);
    if (_skipLayer)
        register_module("linearSkip", _linearSkip);

    if (normalInitializationStd > 0.0) {
        auto* w = _linear2->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
    }
}

torch::Tensor ResNetLinearBlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y = _linear1(leaky_relu(_bn1(x), _reluAlpha));
    y = _linear2(leaky_relu(_bn2(y), _reluAlpha));

    if (_skipLayer)
        x = _linearSkip(x);

    return x + y;
}

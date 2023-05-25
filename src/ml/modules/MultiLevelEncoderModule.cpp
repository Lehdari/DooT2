//
// Project: DooT2
// File: MultiLevelEncoderModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelEncoderModule.hpp"


using namespace ml;
using namespace torch;


MultiLevelEncoderModuleImpl::MultiLevelEncoderModuleImpl(
    double level,
    int inputChannels,
    int outputChannels,
    const ExpandingArray<2>& kernelSize,
    const ExpandingArray<2>& stride
) :
    _level          (level),
    _outputChannels (outputChannels),
    _convMain       (nn::Conv2dOptions(inputChannels, _outputChannels, kernelSize)
                     .stride(stride).bias(false).padding(1)),
    _bnMain         (nn::BatchNorm2dOptions(_outputChannels)),
    _convAux        (nn::Conv2dOptions(3, _outputChannels, {3, 3}).bias(false).padding(1)),
    _bnAux          (nn::BatchNorm2dOptions(_outputChannels))
{
    register_module("convMain", _convMain);
    register_module("bnMain", _bnMain);
    register_module("convAux", _convAux);
    register_module("bnAux", _bnAux);
}

torch::Tensor MultiLevelEncoderModuleImpl::forward(const Tensor& main, const Tensor& aux, double level)
{
    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x;
    if (level > _level) {
        x = torch::leaky_relu(_bnMain(_convMain(main)), leakyReluNegativeSlope);
        torch::Tensor y = torch::leaky_relu(_bnAux(_convAux(aux)), leakyReluNegativeSlope);
        float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
        x = w*y + (1.0f-w)*x;
    }
    else if (level > _level-1.0) {
        x = torch::leaky_relu(_bnAux(_convAux(aux)), leakyReluNegativeSlope);
    }
    else
        x = torch::zeros({aux.sizes()[0], _outputChannels, aux.sizes()[2], aux.sizes()[3]});

    return x;
}

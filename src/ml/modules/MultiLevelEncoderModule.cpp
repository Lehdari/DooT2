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
namespace tf = torch::nn::functional;


MultiLevelEncoderModuleImpl::MultiLevelEncoderModuleImpl(
    double level,
    int inputChannels,
    int outputChannels,
    int xDownScale,
    int yDownScale,
    int resBlockGroups,
    int resBlockScaling
) :
    _level                  (level),
    _outputChannels         (outputChannels),
    _downscaleResBlock      (inputChannels, _outputChannels*resBlockScaling, _outputChannels, xDownScale, yDownScale,
                             resBlockGroups, true, 0.0, 0.001),
    _conv1Aux               (nn::Conv2dOptions(3, _outputChannels, {1, 1}).bias(false)),
    _bn1Aux                 (nn::BatchNorm2dOptions(_outputChannels)),
    _resFourierConvBlock1   (_outputChannels, _outputChannels*resBlockScaling, _outputChannels, resBlockGroups),
    _resFourierConvBlock2   (_outputChannels, _outputChannels*resBlockScaling, _outputChannels, resBlockGroups)
{
    register_module("downscaleResBlock", _downscaleResBlock);
    register_module("conv1Aux", _conv1Aux);
    register_module("bn1Aux", _bn1Aux);
    register_module("resFourierConvBlock1", _resFourierConvBlock1);
    register_module("resFourierConvBlock2", _resFourierConvBlock2);
}

torch::Tensor MultiLevelEncoderModuleImpl::forward(const Tensor& main, const Tensor& aux, double level)
{
    torch::Tensor x, y;
    if (level > _level) {
        x = _downscaleResBlock(main);
        y = gelu(_bn1Aux(_conv1Aux(aux)));
        float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
        x = w*y + (1.0f-w)*x;
        x = _resFourierConvBlock2(_resFourierConvBlock1(x));
    }
    else if (level > _level-1.0) {
        x = gelu(_bn1Aux(_conv1Aux(aux)));
        x = _resFourierConvBlock2(_resFourierConvBlock1(x));
    }
    else
        x = torch::zeros({aux.sizes()[0], _outputChannels, aux.sizes()[2], aux.sizes()[3]},
            torch::TensorOptions().device(main.device()).dtype(main.dtype()));

    return x;
}

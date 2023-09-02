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
    int yDownScale
) :
    _level          (level),
    _outputChannels (outputChannels),
    _conv1Main      (nn::Conv2dOptions(inputChannels, _outputChannels, {yDownScale+2, xDownScale+2})
                     .stride({yDownScale, xDownScale}).bias(false).padding({1, 1})),
    _bn1Main        (nn::BatchNorm2dOptions(_outputChannels)),
    _conv1Aux       (nn::Conv2dOptions(3, _outputChannels, {1, 1}).bias(false)),
    _bn1Aux         (nn::BatchNorm2dOptions(_outputChannels)),
    _resBlock1      (_outputChannels, _outputChannels, _outputChannels, 0.01, 0.001),
    _resBlock2      (_outputChannels, _outputChannels, _outputChannels, 0.01, 0.001)
{
    register_module("conv1Main", _conv1Main);
    register_module("bn1Main", _bn1Main);
    register_module("conv1Aux", _conv1Aux);
    register_module("bn1Aux", _bn1Aux);
    register_module("resBlock1", _resBlock1);
    register_module("resBlock2", _resBlock2);
}

torch::Tensor MultiLevelEncoderModuleImpl::forward(const Tensor& main, const Tensor& aux, double level)
{
    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x, y;
    if (level > _level) {
        x = leaky_relu(_bn1Main(_conv1Main(main)), leakyReluNegativeSlope);
        y = relu(_bn1Aux(_conv1Aux(aux)));
        float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
        x = w*y + (1.0f-w)*x;
        x = _resBlock2(_resBlock1(x));
    }
    else if (level > _level-1.0) {
        x = relu(_bn1Aux(_conv1Aux(aux)));
        x = _resBlock2(_resBlock1(x));
    }
    else
        x = torch::zeros({aux.sizes()[0], _outputChannels, aux.sizes()[2], aux.sizes()[3]},
            torch::TensorOptions().device(main.device()).dtype(main.dtype()));

    return x;
}

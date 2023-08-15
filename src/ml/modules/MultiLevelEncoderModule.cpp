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
    int nGroups1,
    int nGroups2,
    const ExpandingArray<2>& kernelSize,
    const ExpandingArray<2>& stride
) :
    _level          (level),
    _outputChannels (outputChannels),
    _conv1Main      (nn::Conv2dOptions(inputChannels, _outputChannels, kernelSize)
                     .stride(stride).bias(false).padding(1).groups(nGroups1)),
    _bn1Main        (nn::BatchNorm2dOptions(_outputChannels)),
    _conv1Aux       (nn::Conv2dOptions(3, _outputChannels, {3, 3}).bias(false).padding(1)),
    _bn1Aux         (nn::BatchNorm2dOptions(_outputChannels)),
    _conv2          (nn::Conv2dOptions(_outputChannels, _outputChannels, {3, 3})
                     .bias(false).padding(1).groups(nGroups2)),
    _bn2            (nn::BatchNorm2dOptions(_outputChannels)),
    _conv3          (nn::Conv2dOptions(_outputChannels, _outputChannels, {1, 1}).bias(false)),
    _bn3            (nn::BatchNorm2dOptions(_outputChannels))
{
    register_module("conv1Main", _conv1Main);
    register_module("bn1Main", _bn1Main);
    register_module("conv1Aux", _conv1Aux);
    register_module("bn1Aux", _bn1Aux);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    register_module("conv3", _conv3);
    register_module("bn3", _bn3);
}

torch::Tensor MultiLevelEncoderModuleImpl::forward(const Tensor& main, const Tensor& aux, double level)
{
    constexpr double leakyReluNegativeSlope = 0.01;

#if 0
    torch::Tensor x;
    if (level > _level) {
        x = torch::leaky_relu(_bn1Main(_conv1Main(main)), leakyReluNegativeSlope);
        torch::Tensor y = torch::leaky_relu(_bn1Aux(_conv1Aux(aux)), leakyReluNegativeSlope);
        if (this->is_training() && _dropoutRate > 0.0)
            y = tf::dropout(y, tf::DropoutFuncOptions().p(_dropoutRate));
        float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
        x = w*y + (1.0f-w)*x;
    }
    else if (level > _level-1.0) {
        x = torch::leaky_relu(_bn1Aux(_conv1Aux(aux)), leakyReluNegativeSlope);
    }
    else
        x = torch::zeros({aux.sizes()[0], _outputChannels, aux.sizes()[2], aux.sizes()[3]});
#else
    torch::Tensor x = torch::leaky_relu(_bn1Main(_conv1Main(main)), leakyReluNegativeSlope);
    torch::Tensor y;
    float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
    if (w > 0.0) {
        y = torch::leaky_relu(_bn1Aux(_conv1Aux(aux)), leakyReluNegativeSlope);
        x = x + w*y;
    }
    y = _conv3(torch::leaky_relu(_bn2(_conv2(x)), leakyReluNegativeSlope));
    x = torch::leaky_relu(_bn3(x + y), leakyReluNegativeSlope);
#endif
    return x;
}

//
// Project: DooT2
// File: LinearResidualModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/LinearResidualModule.hpp"


using namespace ml;
using namespace torch;


LinearResidualModuleImpl::LinearResidualModuleImpl(int inputOutputSize, int hiddenSize) :
    _bn1        (nn::BatchNorm1dOptions(inputOutputSize)),
    _linear1    (nn::LinearOptions(inputOutputSize, hiddenSize).bias(false)),
    _bn2        (nn::BatchNorm1dOptions(hiddenSize)),
    _linear2    (nn::LinearOptions(hiddenSize, inputOutputSize).bias(false))
{
    register_module("bn1", _bn1);
    register_module("linear1", _linear1);
    register_module("bn2", _bn2);
    register_module("linear2", _linear2);
}

torch::Tensor LinearResidualModuleImpl::forward(torch::Tensor x)
{
    constexpr double leakyReluNegativeSlope = 0.01;

    // full pre-activation, from https://arxiv.org/abs/1603.05027
    torch::Tensor y = _linear1(torch::leaky_relu(_bn1(x), leakyReluNegativeSlope));
    y = _linear2(torch::leaky_relu(_bn2(x), leakyReluNegativeSlope));
    return x+y;
}

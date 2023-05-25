//
// Project: DooT2
// File: Discriminator.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/Discriminator.hpp"


using namespace torch;


ml::DiscriminatorImpl::DiscriminatorImpl() :
    _bn1        (nn::BatchNorm1dOptions(2048)),
    _linear1    (nn::LinearOptions(2048, 32)),
    _bn2        (nn::BatchNorm1dOptions(32)),
    _linear2    (nn::LinearOptions(32, 1))
{
    register_module("encoder", _encoder);
    register_module("bn1", _bn1);
    register_module("linear1", _linear1);
    register_module("bn2", _bn2);
    register_module("linear2", _linear2);

    auto* w = _linear2->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _linear2->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

torch::Tensor ml::DiscriminatorImpl::forward(const MultiLevelImage& img)
{
    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x = torch::leaky_relu(_bn1(_encoder(img)), leakyReluNegativeSlope);
    return 0.5f + torch::tanh(_linear2(torch::leaky_relu(_bn2(_linear1(x)), leakyReluNegativeSlope)))*0.55f;
}

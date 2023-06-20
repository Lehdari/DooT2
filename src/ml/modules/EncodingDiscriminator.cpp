//
// Project: DooT2
// File: EncodingDiscriminator.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/EncodingDiscriminator.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


EncodingDiscriminatorImpl::EncodingDiscriminatorImpl() :
    _conv1      (nn::Conv1dOptions(16, 16, 4).stride(2).padding(1)),
    _conv2      (nn::Conv1dOptions(16, 16, 4).stride(2).padding(1)),
    _conv3      (nn::Conv1dOptions(16, 16, 4).stride(2).padding(1)),
    _conv4      (nn::Conv1dOptions(16, 16, 4).stride(2).padding(1)),
    _conv5      (nn::Conv1dOptions(16, 16, 4).stride(2).padding(1)),
    _linear1    (nn::LinearOptions(1024, 512)),
    _linear2    (nn::LinearOptions(512, 256)),
    _linear3    (nn::LinearOptions(256, 128)),
    _linear4    (nn::LinearOptions(128, 16)),
    _linear5    (nn::LinearOptions(16, 1))
{
    register_module("conv1", _conv1);
    register_module("conv2", _conv2);
    register_module("conv3", _conv3);
    register_module("conv4", _conv4);
    register_module("conv5", _conv5);
    register_module("linear1", _linear1);
    register_module("linear2", _linear2);
    register_module("linear3", _linear3);
    register_module("linear4", _linear4);
    register_module("linear5", _linear5);

    auto* w = _linear5->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.0001);
    w = _linear5->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

torch::Tensor EncodingDiscriminatorImpl::forward(const torch::Tensor& encoding)
{
    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x = torch::reshape(encoding, {1,16,2048});
    x = torch::leaky_relu(_conv1(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_conv2(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_conv3(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_conv4(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_conv5(x), leakyReluNegativeSlope);
    x = torch::reshape(x, {1,1024});
    x = torch::leaky_relu(_linear1(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_linear2(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_linear3(x), leakyReluNegativeSlope);
    x = torch::leaky_relu(_linear4(x), leakyReluNegativeSlope);
    return _linear5(x);
}

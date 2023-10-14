//
// Project: DooT2
// File: AdaptiveConv2d.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/AdaptiveConv2d.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


namespace {

inline int contextMappingHiddenChannels(int contextChannels, int filterBankSize)
{
    return (int)std::pow(2.0, std::ceil((std::log2((double)contextChannels) +
        std::log2((double)filterBankSize)) * 0.5));
}

} // namespace


AdaptiveConv2dImpl::AdaptiveConv2dImpl(
    int inputChannels,
    int outputChannels,
    int contextChannels,
    const std::vector<long>& kernelSize,
    int groups,
    int filterBankSize,
    const std::vector<long>& padding,
    double normalInitializationStd,
    bool useReflectionPadding
):
    _weight         (torch::zeros({filterBankSize, outputChannels, inputChannels/groups, kernelSize[0],
                     kernelSize[1]})),
    _linear1        (nn::LinearOptions(contextChannels, contextMappingHiddenChannels(contextChannels,
                     filterBankSize))),
    _linear2        (nn::LinearOptions(contextMappingHiddenChannels(contextChannels, filterBankSize),
                     filterBankSize)),
    _groups         (groups),
    _padding        (padding),
    _paddingMode    (*std::max_element(_padding.begin(), _padding.end()) == 0 ? 0 :
                     (useReflectionPadding ? 2 : 1))
{
    if (normalInitializationStd > 0.0) {
        torch::nn::init::normal_(_weight, 0.0, normalInitializationStd);
    }
    else {
        double stdv = 1.0 / std::sqrt(inputChannels);
        torch::nn::init::uniform_(_weight, -stdv, stdv);
    }
    register_parameter("weight", _weight);
    register_module("linear1", _linear1);
    register_module("linear2", _linear2);
}

torch::Tensor AdaptiveConv2dImpl::forward(torch::Tensor x, const torch::Tensor& context)
{
    auto b = x.sizes()[0]; // batch size
    auto c = x.sizes()[1]; // input channels
    auto o = _weight.sizes()[1]; // output channels

    switch (_paddingMode) {
        case 1:
            x = torch::pad(x, _padding);
            break;
        case 2:
            x = torch::reflection_pad2d(x, _padding);
            break;
        default:
            break;
    }

    // Filter soft-selection weights with 2-layer MLP and softmax
    torch::Tensor y = tf::softmax(_linear2(gelu(_linear1(context), "tanh")), tf::SoftmaxFuncOptions(1));

    assert(y.sizes()[0] == b);
    auto f = y.sizes()[1]; // filter bank size

    // Select filters from batch by multiplication-summation:
    // _weight and y are both casted to 6D ("B x F x O x I x Kh x Kw") and summed along F
    // to produce a filter for each entry in the batch.
    torch::Tensor weight = (_weight.unsqueeze(0) * y.view({b, f, 1, 1, 1, 1})).sum(1);

    // Batched convolution is not supported so weight and input are both converted so that batch dimensionality
    // is moved to the channels. The batches are kept separate with the groups (hence b*_groups) and casted
    // back to expected batch size and n. of channels after the convolution.
    weight = weight.view({b*o, _weight.sizes()[2], _weight.sizes()[3], _weight.sizes()[4]});
    x = tf::conv2d(x.view({1, b*c, x.sizes()[2], x.sizes()[3]}), weight, tf::Conv2dFuncOptions().groups(b*_groups));
    x = x.view({b, o, x.sizes()[2], x.sizes()[3]});

    return x;
}

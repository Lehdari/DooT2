//
// Project: DooT2
// File: AdaptiveConvTranspose2d.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/AdaptiveConvTranspose2d.hpp"


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


AdaptiveConvTranspose2dImpl::AdaptiveConvTranspose2dImpl(
    int inputChannels,
    int outputChannels,
    int contextChannels,
    int xUpscale,
    int yUpscale,
    int groups,
    int filterBankSize,
    double normalInitializationStd
):
    _xUpScale       (xUpscale),
    _yUpScale       (yUpscale),
    _groups         (groups),
    _weight         (torch::zeros({filterBankSize, inputChannels, outputChannels/groups, yUpscale+2, xUpscale+2})),
    _linear1a       (nn::LinearOptions(contextChannels, contextMappingHiddenChannels(contextChannels,
                     filterBankSize))),
    _linear2a       (nn::LinearOptions(contextMappingHiddenChannels(contextChannels, filterBankSize),
                     filterBankSize)),
    _linear1b       (nn::LinearOptions(contextChannels, inputChannels)),
    _linear2b       (nn::LinearOptions(inputChannels, inputChannels))
{
    if (normalInitializationStd > 0.0) {
        torch::nn::init::normal_(_weight, 0.0, normalInitializationStd);
    }
    else {
        double stdv = 1.0 / std::sqrt(inputChannels);
        torch::nn::init::uniform_(_weight, -stdv, stdv);
    }
    register_parameter("weight", _weight);
    register_module("linear1a", _linear1a);
    register_module("linear2a", _linear2a);
    register_module("linear1b", _linear1b);
    register_module("linear2b", _linear2b);
}

torch::Tensor AdaptiveConvTranspose2dImpl::forward(torch::Tensor x, const torch::Tensor& context)
{
    using namespace torch::indexing;

    auto b = x.sizes()[0]; // batch size
    auto c = x.sizes()[1]; // input channels
    auto o = _weight.sizes()[2]*_groups; // output channels

    // Filter soft-selection weights with 2-layer MLP and softmax
    torch::Tensor y = tf::softmax(_linear2a(gelu(_linear1a(context), "tanh")), tf::SoftmaxFuncOptions(1));
    assert(y.sizes()[0] == b);

    // Filter modulation weights
    torch::Tensor z = torch::sigmoid(_linear2b(gelu(_linear1b(context), "tanh")));
    assert(z.sizes()[0] == b);

    auto f = y.sizes()[1]; // filter bank size

    // Select filters from batch by multiplication-summation:
    // _weight and y are both casted to 6D ("B x F x I x O x Kh x Kw") and summed along F
    // to produce a filter for each entry in the batch.
    torch::Tensor weight = (_weight.unsqueeze(0) * y.view({b, f, 1, 1, 1, 1})).sum(1);

    // filter weight modulation along input dimension ("B x I x O x Kh x Kw")
    weight = weight * z.view({b, c, 1, 1, 1});

    // Batched convolution is not supported so weight and input are both converted so that batch dimensionality
    // is moved to the channels. The batches are kept separate with the groups (hence b*_groups) and casted
    // back to expected batch size and n. of channels after the convolution.
    weight = weight.view({b*c, _weight.sizes()[2], _weight.sizes()[3], _weight.sizes()[4]});
    x = tf::conv_transpose2d(x.view({1, b*c, x.sizes()[2], x.sizes()[3]}), weight, tf::ConvTranspose2dFuncOptions()
        .groups(b*_groups).stride({_yUpScale, _xUpScale}));
    x = x.view({b, o, x.sizes()[2], x.sizes()[3]});

    return x.index({Slice(), Slice(), Slice(1,-1), Slice(1,-1)});
}

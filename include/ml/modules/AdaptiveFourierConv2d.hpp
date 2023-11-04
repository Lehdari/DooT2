//
// Project: DooT2
// File: AdaptiveFourierConv2d.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/modules/AdaptiveConv2d.hpp"


namespace ml {

class AdaptiveFourierConv2dImpl : public torch::nn::Module {
public:
    explicit AdaptiveFourierConv2dImpl(
        int inputChannels,
        int outputChannels,
        int contextChannels,
        double globalChannelRatio = 0.5,
        int groups = 1,
        int filterBankSize = 16,
        double normalInitializationStd = 0.0,
        bool useReflectionPadding = false // false: zero padding
    );

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& context);

private:
    int                     _localInputChannels;
    int                     _globalInputChannels;
    int                     _localOutputChannels;
    int                     _globalOutputChannels;

    AdaptiveConv2d          _convLocal;
    AdaptiveConv2d          _convLocalGlobal;
    AdaptiveConv2d          _convGlobalLocal;
    torch::nn::Conv2d       _convGlobal1;
    AdaptiveConv2d          _convGlobal2;
    torch::nn::Conv2d       _convGlobal3;
    torch::nn::BatchNorm2d  _bnGlobal1;
    torch::nn::BatchNorm2d  _bnGlobal2;
    torch::nn::BatchNorm2d  _bnLocalMerge;
    torch::nn::BatchNorm2d  _bnGlobalMerge;
};
TORCH_MODULE(AdaptiveFourierConv2d);

} // namespace ml

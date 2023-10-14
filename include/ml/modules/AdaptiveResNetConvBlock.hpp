//
// Project: DooT2
// File: AdaptiveResNetConvBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>
#include "ml/modules/AdaptiveConv2d.hpp"
#include "ml/modules/AdaptiveSqueezeExcitation.hpp"


namespace ml {

class AdaptiveResNetConvBlockImpl : public torch::nn::Module {
public:
    explicit AdaptiveResNetConvBlockImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels,
        int contextChannels,
        int groups = 1,
        int filterBankSize = 16,
        bool useSqueezeExcitation = false,
        double normalInitializationStd = 0.0,
        bool useReflectionPadding = false // false: zero padding
    );

    torch::Tensor forward(torch::Tensor x, const torch::Tensor& context);

private:
    bool                        _skipLayer; // true if input size differs from output size, so the adapter layer is required
    bool                        _useSqueezeExcitation;

    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn1;
    AdaptiveConv2d              _conv2;
    torch::nn::BatchNorm2d      _bn2;
    AdaptiveSqueezeExcitation   _se1;
    torch::nn::Conv2d           _conv3;
    torch::nn::Conv2d           _convSkip;
};
TORCH_MODULE(AdaptiveResNetConvBlock);

} // namespace ml

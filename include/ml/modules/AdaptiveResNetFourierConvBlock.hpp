//
// Project: DooT2
// File: AdaptiveResNetFourierConvBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/modules/AdaptiveFourierConv2d.hpp"


namespace ml {

class AdaptiveResNetFourierConvBlockImpl : public torch::nn::Module {
public:
    explicit AdaptiveResNetFourierConvBlockImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels,
        int contextChannels,
        int groups = 1,
        int filterBankSize = 16,
        double normalInitializationStd = 0.0
    );

    torch::Tensor forward(torch::Tensor x, const torch::Tensor& context);

private:
    bool                _skipLayer; // true if input size differs from output size, so the adapter layer is required

    AdaptiveFourierConv2d   _fourierConv1;
    AdaptiveFourierConv2d   _fourierConv2;
    torch::nn::Conv2d       _convSkip;
};
TORCH_MODULE(AdaptiveResNetFourierConvBlock);

} // namespace ml

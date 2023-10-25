//
// Project: DooT2
// File: ResNetFourierConvBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/modules/FourierConv2d.hpp"


namespace ml {

class ResNetFourierConvBlockImpl : public torch::nn::Module {
public:
    explicit ResNetFourierConvBlockImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels,
        int groups = 1,
        double normalInitializationStd = 0.0
    );

    torch::Tensor forward(torch::Tensor x);

private:
    bool                _skipLayer; // true if input size differs from output size, so the adapter layer is required

    FourierConv2d       _fourierConv1;
    FourierConv2d       _fourierConv2;
    torch::nn::Conv2d   _convSkip;
};
TORCH_MODULE(ResNetFourierConvBlock);

} // namespace ml

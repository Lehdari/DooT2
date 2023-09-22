//
// Project: DooT2
// File: ResNetDownscaleConvBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>
#include "ml/modules/SqueezeExcitation.hpp"


namespace ml {

    class ResNetDownscaleConvBlockImpl : public torch::nn::Module {
    public:
        explicit ResNetDownscaleConvBlockImpl(
            int inputChannels,
            int hiddenChannels,
            int outputChannels,
            int xDownScale,
            int yDownScale,
            int groups = 1,
            bool useSqueezeExcitation = false,
            double normalInitializationStd = 0.0,
            double skipNormalInitializationStd = 0.0
        );

        torch::Tensor forward(torch::Tensor x);

    private:
        bool                        _skipLayer; // true if input size differs from output size, so the adapter layer is required
        bool                        _useSqueezeExcitation;

        torch::nn::Conv2d           _conv1;
        torch::nn::BatchNorm2d      _bn1;
        torch::nn::Conv2d           _conv2;
        torch::nn::BatchNorm2d      _bn2;
        SqueezeExcitation           _se1;
        torch::nn::Conv2d           _conv3;
        torch::nn::AvgPool2d        _avgPool1;
        torch::nn::Conv2d           _convSkip;
    };
    TORCH_MODULE(ResNetDownscaleConvBlock);

} // namespace ml

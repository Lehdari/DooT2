//
// Project: DooT2
// File: ResNetConvBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class ResNetConvBlockImpl : public torch::nn::Module {
public:
    explicit ResNetConvBlockImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels,
        int groups1 = 1,
        int groups2 = 1,
        double reluAlpha=0.01,
        double normalInitializationStd=0.0
    );

    torch::Tensor forward(torch::Tensor x);

private:
    double                      _reluAlpha;
    bool                        _skipLayer; // true if input size differs from output size, so the adapter layer is required

    torch::nn::BatchNorm2d      _bn1;
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn2;
    torch::nn::Conv2d           _conv2;
    torch::nn::Conv2d           _convSkip;
};
TORCH_MODULE(ResNetConvBlock);

} // namespace ml

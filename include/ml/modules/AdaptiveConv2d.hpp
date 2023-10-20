//
// Project: DooT2
// File: AdaptiveConv2d.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class AdaptiveConv2dImpl : public torch::nn::Module {
public:
    explicit AdaptiveConv2dImpl(
        int inputChannels,
        int outputChannels,
        int contextInputChannels,
        const std::vector<long>& kernelSize,
        int groups = 1,
        int filterBankSize = 16,
        const std::vector<long>& padding = {0,0,0,0},
        double normalInitializationStd = 0.0,
        bool useReflectionPadding = false // false: zero padding
    );

    torch::Tensor forward(torch::Tensor x, const torch::Tensor& context);

private:
    torch::Tensor               _weight;
    torch::nn::Linear           _linear1a; // two linear layers for filter selection with context
    torch::nn::Linear           _linear2a;
    torch::nn::Linear           _linear1b; // two linear layers for filter modulation with context
    torch::nn::Linear           _linear2b;

    int                         _groups;
    const std::vector<long>     _padding;
    int                         _paddingMode; // 0: no padding, 1: zero padding, 2: reflection padding
};
TORCH_MODULE(AdaptiveConv2d);

} // namespace ml

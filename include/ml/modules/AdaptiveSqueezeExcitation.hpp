//
// Project: DooT2
// File: AdaptiveSqueezeExcitation.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class AdaptiveSqueezeExcitationImpl : public torch::nn::Module {
public:
    explicit AdaptiveSqueezeExcitationImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels,
        int contextChannels
    );

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& context);

private:
    torch::nn::AdaptiveAvgPool2d    _avgPool1;
    torch::nn::Linear               _linear1a;
    torch::nn::Linear               _linear1b;
    torch::nn::Linear               _linear2;
};
TORCH_MODULE(AdaptiveSqueezeExcitation);

} // namespace ml

//
// Project: DooT2
// File: SqueezeExcitation.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class SqueezeExcitationImpl : public torch::nn::Module {
public:
    explicit SqueezeExcitationImpl(
        int inputChannels,
        int hiddenChannels,
        int outputChannels
    );

    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::AdaptiveAvgPool2d    _avgPool1;
    torch::nn::Linear               _linear1;
    torch::nn::Linear               _linear2;
};
TORCH_MODULE(SqueezeExcitation);

} // namespace ml

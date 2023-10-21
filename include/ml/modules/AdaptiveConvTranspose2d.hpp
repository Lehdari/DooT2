//
// Project: DooT2
// File: AdaptiveConvTranspose2d.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class AdaptiveConvTranspose2dImpl : public torch::nn::Module {
public:
    explicit AdaptiveConvTranspose2dImpl(
        int inputChannels,
        int outputChannels,
        int contextInputChannels,
        int xUpscale,
        int yUpscale,
        int groups = 1,
        int filterBankSize = 16,
        double normalInitializationStd = 0.0
    );

    torch::Tensor forward(torch::Tensor x, const torch::Tensor& context);

private:
    int                         _xUpScale;
    int                         _yUpScale;
    int                         _groups;

    torch::Tensor               _weight;
    torch::nn::Linear           _linear1a; // two linear layers for filter selection with context
    torch::nn::Linear           _linear2a;
    torch::nn::Linear           _linear1b; // two linear layers for filter modulation with context
    torch::nn::Linear           _linear2b;
};
TORCH_MODULE(AdaptiveConvTranspose2d);

} // namespace ml

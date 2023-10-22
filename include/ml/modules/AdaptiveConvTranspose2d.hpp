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
        const std::vector<long>& kernelSize,
        int groups = 1,
        int filterBankSize = 16,
        const std::vector<long>& stride = {1,1},
        const std::vector<long>& cropping = {0,0,0,0}, // top, bottom, left, right crop on output
        double normalInitializationStd = 0.0
    );

    torch::Tensor forward(torch::Tensor x, const torch::Tensor& context);

private:
    int                         _groups;
    std::vector<long>           _stride;
    std::vector<long>           _cropping;

    torch::Tensor               _weight;
    torch::nn::Linear           _linear1a; // two linear layers for filter selection with context
    torch::nn::Linear           _linear2a;
    torch::nn::Linear           _linear1b; // two linear layers for filter modulation with context
    torch::nn::Linear           _linear2b;
};
TORCH_MODULE(AdaptiveConvTranspose2d);

} // namespace ml

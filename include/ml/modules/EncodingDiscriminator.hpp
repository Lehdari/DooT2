//
// Project: DooT2
// File: EncodingDiscriminator.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/modules/LinearResidualModule.hpp"

#include <torch/torch.h>


namespace ml {

class EncodingDiscriminatorImpl : public torch::nn::Module {
public:
    EncodingDiscriminatorImpl();

    // Expects tensor of shape (1, B, N), returns scalar
    torch::Tensor forward(const torch::Tensor& encoding);

private:
    torch::nn::Conv1d       _conv1;
    torch::nn::Conv1d       _conv2;
    torch::nn::Conv1d       _conv3;
    torch::nn::Conv1d       _conv4;
    torch::nn::Conv1d       _conv5;
    torch::nn::Linear       _linear1;
    torch::nn::Linear       _linear2;
    torch::nn::Linear       _linear3;
    torch::nn::Linear       _linear4;
    torch::nn::Linear       _linear5;
};
TORCH_MODULE(EncodingDiscriminator);

} // namespace ml
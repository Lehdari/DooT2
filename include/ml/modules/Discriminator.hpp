//
// Project: DooT2
// File: Discriminator.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "MultiLevelFrameEncoder.hpp"

#include <torch/torch.h>


namespace ml {

class DiscriminatorImpl : public torch::nn::Module {
public:
    DiscriminatorImpl();

    torch::Tensor forward(const MultiLevelImage& img);

private:
    MultiLevelFrameEncoder  _encoder;
    torch::nn::BatchNorm1d  _bn1;
    torch::nn::Linear       _linear1;
    torch::nn::BatchNorm1d  _bn2;
    torch::nn::Linear       _linear2;
};
TORCH_MODULE(Discriminator);

} // namespace ml

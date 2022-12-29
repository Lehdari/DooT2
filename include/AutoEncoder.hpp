//
// Project: DooT2
// File: AutoEncoder.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


class AutoEncoderImpl : public torch::nn::Module {
public:
    AutoEncoderImpl();

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d           _conv;
    torch::nn::ConvTranspose2d  _convTranspose;
};
TORCH_MODULE(AutoEncoder);

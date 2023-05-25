//
// Project: DooT2
// File: MultiLevelEncoderModule.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/MultiLevelImage.hpp"

#include <torch/torch.h>


namespace ml {

class MultiLevelEncoderModuleImpl : public torch::nn::Module {
public:
    MultiLevelEncoderModuleImpl(
        double level,
        int inputChannels,
        int outputChannels,
        const torch::ExpandingArray<2>& kernelSize,
        const torch::ExpandingArray<2>& stride
    );

    torch::Tensor forward(const torch::Tensor& main, const torch::Tensor& aux, double level);

private:
    double                  _level;
    int                     _outputChannels;
    torch::nn::Conv2d       _convMain; // layers for the primary feedforward
    torch::nn::BatchNorm2d  _bnMain;
    torch::nn::Conv2d       _convAux; // layers for the downscaled secondary input
    torch::nn::BatchNorm2d  _bnAux;
};
TORCH_MODULE(MultiLevelEncoderModule);

} // namespace ml

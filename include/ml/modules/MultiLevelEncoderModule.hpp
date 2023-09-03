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
#include "ml/modules/ResNetConvBlock.hpp"

#include <torch/torch.h>


namespace ml {

class MultiLevelEncoderModuleImpl : public torch::nn::Module {
public:
    MultiLevelEncoderModuleImpl(
        double level,
        int inputChannels,
        int outputChannels,
        int xDownScale,
        int yDownScale,
        int resBlockGroups = 1
    );

    torch::Tensor forward(const torch::Tensor& main, const torch::Tensor& aux, double level);

private:
    double                  _level;
    int                     _outputChannels;
    torch::nn::Conv2d       _conv1Main; // layers for the primary feedforward
    torch::nn::BatchNorm2d  _bn1Main;
    torch::nn::Conv2d       _conv1Aux; // layers for the downscaled secondary input
    torch::nn::BatchNorm2d  _bn1Aux;
    ResNetConvBlock         _resBlock1;
    ResNetConvBlock         _resBlock2;
};
TORCH_MODULE(MultiLevelEncoderModule);

} // namespace ml

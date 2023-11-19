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
#include "ml/modules/ResNetDownscaleConvBlock.hpp"

#include <torch/torch.h>


namespace ml {

class MultiLevelEncoderModuleImpl : public torch::nn::Module {
public:
    MultiLevelEncoderModuleImpl(
        const std::string& resBlockConfig, // Composition of residual blocks. T: transformer, C: convolution, F: fourier convolution
        double level,
        int inputChannels,
        int outputChannels,
        int xDownScale,
        int yDownScale,
        int resBlockGroups = 1,
        int resBlockScaling = 1,
        int transformerHeads = 16,
        int transformerHeadDim = 32
    );

    torch::Tensor forward(const torch::Tensor& main, const torch::Tensor& aux, double level);

private:
    std::string                 _resBlockConfig;
    double                      _level;
    int                         _outputChannels;
    ResNetDownscaleConvBlock    _downscaleResBlock;
    torch::nn::Conv2d           _conv1Aux; // layers for the downscaled secondary input
    torch::nn::BatchNorm2d      _bn1Aux;
    torch::nn::ModuleList       _resBlocks;
};
TORCH_MODULE(MultiLevelEncoderModule);

} // namespace ml

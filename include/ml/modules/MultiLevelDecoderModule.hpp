//
// Project: DooT2
// File: MultiLevelDecoderModule.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/MultiLevelImage.hpp"
#include "ml/modules/AdaptiveConvTranspose2d.hpp"

#include <torch/torch.h>


namespace ml {

class MultiLevelDecoderModuleImpl : public torch::nn::Module {
public:
    MultiLevelDecoderModuleImpl(
        const std::string& resBlockConfig, // Composition of residual blocks. T: transformer, C: convolution, F: fourier convolution
        double level,
        int inputChannels,
        int outputChannels,
        int contextChannels,
        int xUpscale,
        int yUpscale,
        int upscaleConvGroups = 1,
        int resBlockGroups = 1,
        int resBlockScaling = 1,
        int filterBankSize = 16,
        int transformerHeads = 16,
        int transformerHeadDim = 32
    );

    // outputs tuple of main tensor, auxiliary image
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x,
        const torch::Tensor& context,
        double level,
        const torch::Tensor* imgPrev = nullptr
    );

private:
    std::string                     _resBlockConfig;
    double                          _level;
    int                             _outputChannels;
    int                             _xUpScale;
    int                             _yUpScale;
    AdaptiveConvTranspose2d         _convTranspose1;
    torch::nn::ModuleList           _resBlocks;
    torch::nn::Conv2d               _convAux;
    torch::nn::BatchNorm2d          _bnAux;
    torch::nn::Conv2d               _conv_Y;
    torch::nn::Conv2d               _conv_UV;
};
TORCH_MODULE(MultiLevelDecoderModule);

} // namespace ml

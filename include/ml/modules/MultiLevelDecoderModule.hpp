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
#include "ml/modules/ResNetConvBlock.hpp"

#include <torch/torch.h>


namespace ml {

    class MultiLevelDecoderModuleImpl : public torch::nn::Module {
    public:
        MultiLevelDecoderModuleImpl(
            double level,
            int inputChannels,
            int outputChannels,
            int xUpscale,
            int yUpscale,
            int resBlockGroups = 1,
            int resBlockScaling = 1
        );

        // outputs tuple of main tensor, auxiliary image
        std::tuple<torch::Tensor, torch::Tensor> forward(
            torch::Tensor x,
            double level,
            const torch::Tensor* imgPrev = nullptr
        );

    private:
        double                  _level;
        int                     _outputChannels;
        int                     _xUpScale;
        int                     _yUpScale;
        ResNetConvBlock         _resBlock1;
        ResNetConvBlock         _resBlock2;
        torch::nn::Conv2d       _convAux;
        torch::nn::BatchNorm2d  _bnAux;
        torch::nn::Conv2d       _conv_Y;
        torch::nn::Conv2d       _conv_UV;
    };
    TORCH_MODULE(MultiLevelDecoderModule);

} // namespace ml

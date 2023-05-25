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
#include "ml/modules/ResNeXtModule.hpp"

#include <torch/torch.h>


namespace ml {

    class MultiLevelDecoderModuleImpl : public torch::nn::Module {
    public:
        MultiLevelDecoderModuleImpl(
            double level,
            int inputChannels,
            int hiddenChannels1,
            int hiddenChannels2,
            int outputChannels,
            int nGroups1,
            int nGroups2,
            int groupChannels1,
            int groupChannels2,
            int outputWidth,
            int outputHeight,
            const torch::ExpandingArray<2>& kernelSize,
            const torch::ExpandingArray<2>& stride = {1,1},
            const torch::indexing::Slice& vSlice = torch::indexing::Slice(),
            const torch::indexing::Slice& hSlice = torch::indexing::Slice()
        );

        // outputs tuple of main tensor, auxiliary image
        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, double level);

    private:
        double                      _level;
        int                         _outputWidth;
        int                         _outputHeight;
        torch::indexing::Slice      _vSlice;
        torch::indexing::Slice      _hSlice;
        ResNeXtModule               _resNext1;
        ResNeXtModule               _resNext2;
        torch::nn::ConvTranspose2d  _convTranspose; // layers for the primary feedforward
        torch::nn::BatchNorm2d      _bnMain;
        torch::nn::Conv2d           _convAux;
        torch::nn::BatchNorm2d      _bnAux;
        torch::nn::Conv2d           _conv_Y;
        torch::nn::Conv2d           _conv_UV;
    };
    TORCH_MODULE(MultiLevelDecoderModule);

} // namespace ml

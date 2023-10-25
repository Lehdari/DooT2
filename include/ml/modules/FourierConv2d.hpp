//
// Project: DooT2
// File: FourierConv2d.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

// Fast Fourier Convolution
// https://papers.nips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
class FourierConv2dImpl : public torch::nn::Module {
public:
    explicit FourierConv2dImpl(
        int inputChannels,
        int outputChannels,
        double globalChannelRatio = 0.5,
        int groups = 1,
        double normalInitializationStd = 0.0
    );

    torch::Tensor forward(const torch::Tensor& x);

private:
    int                     _localInputChannels;
    int                     _globalInputChannels;
    int                     _localOutputChannels;
    int                     _globalOutputChannels;

    torch::nn::Conv2d       _convLocal;
    torch::nn::Conv2d       _convLocalGlobal;
    torch::nn::Conv2d       _convGlobalLocal;
    torch::nn::Conv2d       _convGlobal1;
    torch::nn::Conv2d       _convGlobal2;
    torch::nn::Conv2d       _convGlobal3;
    torch::nn::BatchNorm2d  _bnGlobal1;
    torch::nn::BatchNorm2d  _bnGlobal2;
    torch::nn::BatchNorm2d  _bnLocalMerge;
    torch::nn::BatchNorm2d  _bnGlobalMerge;
};
TORCH_MODULE(FourierConv2d);

} // namespace ml

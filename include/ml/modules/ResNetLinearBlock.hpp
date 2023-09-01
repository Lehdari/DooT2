//
// Project: DooT2
// File: ResNetLinearBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class ResNetLinearBlockImpl : public torch::nn::Module {
public:
    explicit ResNetLinearBlockImpl(int inputChannels, int hiddenChannels, int outputChannels, double reluAlpha=0.01);

    torch::Tensor forward(torch::Tensor x);

private:
    double                      _reluAlpha;
    bool                        _skipLayer;

    torch::nn::BatchNorm1d      _bn1;
    torch::nn::Linear           _linear1;
    torch::nn::BatchNorm1d      _bn2;
    torch::nn::Linear           _linear2;
    torch::nn::Linear           _linearSkip;
};
TORCH_MODULE(ResNetLinearBlock);

} // namespace ml

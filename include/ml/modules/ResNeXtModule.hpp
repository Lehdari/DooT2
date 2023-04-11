//
// Project: DooT2
// File: ResNeXtModule.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

class ResNeXtModuleImpl : public torch::nn::Module {
public:
    ResNeXtModuleImpl(int nInputChannels, int nOutputChannels, int nGroups, int nGroupChannels = -1);

    torch::Tensor forward(torch::Tensor x);

private:
    int                         _nInputChannels;
    int                         _nOutputChannels;
    int                         _nGroups;
    int                         _nGroupChannels;

    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn1;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bn2;
    torch::nn::Conv2d           _conv3;
    torch::nn::BatchNorm2d      _bn3;

    std::vector<torch::Tensor>  _groupOutputs;
};
TORCH_MODULE(ResNeXtModule);

} // namespace ml

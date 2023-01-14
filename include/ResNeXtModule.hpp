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


class ResNeXtModuleImpl : public torch::nn::Module {
public:
    // only nInputChannels == nOutputChannels supported for now
    ResNeXtModuleImpl(int nInputChannels, int nGroupChannels, int nGroups, int nOutputChannels);

    torch::Tensor forward(torch::Tensor x);

private:
    int                                 _nGroups;

    std::vector<torch::nn::Sequential>  _groups;
    torch::nn::Conv2d                   _convFinal;
    torch::nn::BatchNorm2d              _bnFinal;

    std::vector<torch::Tensor>          _groupOutputs;
};
TORCH_MODULE(ResNeXtModule);

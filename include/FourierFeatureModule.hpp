//
// Project: DooT2
// File: FourierFeatureModule.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

class FourierFeatureModuleImpl : public torch::nn::Module {
public:
    FourierFeatureModuleImpl(int nInputChannels, int nFeatures, double minStd = 1.0, double maxStd = 1.0);

    torch::Tensor forward(torch::Tensor x);

private:
    int                 _nFeatures;
    torch::nn::Conv2d   _conv;
};
TORCH_MODULE(FourierFeatureModule);
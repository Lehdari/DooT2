//
// Project: DooT2
// File: LinearResidualModule.hpp
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

class LinearResidualModuleImpl : public torch::nn::Module {
public:
    LinearResidualModuleImpl(int inputOutputSize, int hiddenSize);

    // outputs tuple of main tensor, auxiliary image
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::BatchNorm1d      _bn1;
    torch::nn::Linear           _linear1;
    torch::nn::BatchNorm1d      _bn2;
    torch::nn::Linear           _linear2;
};
TORCH_MODULE(LinearResidualModule);

} // namespace ml


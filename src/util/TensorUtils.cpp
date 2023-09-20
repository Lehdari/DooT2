//
// Project: DooT2
// File: TensorUtils.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "util/TensorUtils.hpp"


// Adapted from https://github.com/lucidrains/vit-pytorch/blob/8208c859a5474b2d93b429202833fcd9f395ec30/vit_pytorch/simple_vit.py#L12
torch::Tensor positionalEmbedding2D(int h, int w, int dim, double temperature)
{
    using namespace torch::indexing;

    auto yx = torch::meshgrid({torch::arange(h), torch::arange(w)}, "ij");
    auto& y = yx.at(0);
    auto& x = yx.at(1);

    assert(dim % 4 == 0);
    torch::Tensor omega = torch::arange(dim/4) / (double)(dim/4 - 1);
    omega = 1.0 / (torch::pow(temperature, omega));

    y = y.flatten().index({Slice(), None}) * omega.index({None, Slice()});
    x = x.flatten().index({Slice(), None}) * omega.index({None, Slice()});

    return torch::cat({x.sin(), x.cos(), y.sin(), y.cos()}, dim=1);
}
//
// Project: DooT2
// File: TransformerBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/TransformerBlock.hpp"


using namespace ml;
using namespace torch;


TransformerBlockImpl::TransformerBlockImpl(int dim, int heads, int headDim, int linearDim) :
    _attention1             (dim, heads, headDim),
    _ln1                    (nn::LayerNormOptions({dim})),
    _linear1                (nn::LinearOptions(dim, linearDim)),
    _linear2                (nn::LinearOptions(linearDim, dim))
{
    register_module("attention1", _attention1);
    register_module("ln1", _ln1);
    register_module("linear1", _linear1);
    register_module("linear2", _linear2);
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x)
{
    // Attention
    x = x + _attention1(x);
    // MLP
    x = x + _linear2(torch::silu(_linear1(_ln1(x))));

    return x;
}

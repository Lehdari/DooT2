//
// Project: DooT2
// File: ViTBlock.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ViTBlock.hpp"
#include "util/TensorUtils.hpp"


using namespace ml;
using namespace torch;


ViTBlockImpl::ViTBlockImpl(int dim, int heads, int headDim, int linearDim, bool usePositionalEmbedding) :
    _usePositionalEmbedding (usePositionalEmbedding),
    _ln1            (nn::LayerNormOptions({dim})),
    _transformer1   (dim, heads, headDim, linearDim)
{
    register_module("ln1", _ln1);
    register_module("transformer1", _transformer1);
}

torch::Tensor ViTBlockImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    int b = x.sizes()[0];
    int c = x.sizes()[1];
    int h = x.sizes()[2];
    int w = x.sizes()[3];
    auto dtype = x.dtype();
    auto device = x.device();

    // 2D to 1D
    x = torch::reshape(torch::permute(x, {0, 2, 3, 1}), {b, h*w, c}); // BSC (c == dim)

    // Normalization
    x = _ln1(x).to(dtype).to(device);

    // Positional embedding
    if (_usePositionalEmbedding) {
        if (_posEmbedding.sizes()[0] != w*h)
            _posEmbedding = positionalEmbedding2D(h, w, c).to(dtype).to(device);
        x = x + _posEmbedding;
    }

    // Transformer block
    x = _transformer1(x);

    // 1D to 2D
    x = torch::reshape(torch::permute(x, {0, 2, 1}), {b, c, h, w});

    return x;
}

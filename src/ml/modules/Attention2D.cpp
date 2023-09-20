//
// Project: DooT2
// File: Attention2D.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/Attention2D.hpp"
#include "util/TensorUtils.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


Attention2DImpl::Attention2DImpl(int dim, int heads, int headDim) :
    _heads          (heads),
    _innerDim       (headDim * heads),
    _scale          (std::pow((double)headDim, -0.5)),
    _ln1            (nn::LayerNormOptions({dim})),
    _linear1        (nn::LinearOptions(dim, _innerDim*3)),
    _softmax1       (nn::SoftmaxOptions(-1)),
    _linear2        (nn::LinearOptions(_innerDim, dim))
{
    register_module("ln1", _ln1);
    register_module("linear1", _linear1);
    register_module("softmax1", _softmax1);
    register_module("linear2", _linear2);
}

torch::Tensor Attention2DImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    int b = x.sizes()[0];
    int c = x.sizes()[1];
    int h = x.sizes()[2];
    int w = x.sizes()[3];

    x = torch::reshape(torch::permute(x, {0, 2, 3, 1}), {b, h*w, c}); // BSC (c == dim)

    // Normalization
    x = _ln1(x);

    // Positional embedding
    if (_posEmbedding.sizes()[0] != w*h)
        _posEmbedding = positionalEmbedding2D(h, w, c).to(x.dtype()).to(x.device());
    x = x + _posEmbedding;

    // Produce queries, keys and values
    x = _linear1(x);
    x = torch::permute(torch::reshape(x, {b, h*w, 3, _heads, -1}), {2, 0, 3, 1, 4}); // 3BHNS (H: heads)
    Tensor q = x.index({0});
    Tensor k = x.index({1});
    Tensor v = x.index({2});

    // Dot product attention
    x = torch::matmul(q, k.transpose(-1, -2)) * _scale;
    x = _softmax1(x);

    // Gather attention-weighed output
    x = torch::matmul(x, v);

    // Output linear layer and final reshape
    x = torch::reshape(torch::permute(x, {0, 2, 1, 3}), {b, w*h, _innerDim}); // BCS
    x = _linear2(x);
    x = torch::reshape(torch::permute(x, {0, 2, 1}), {b, c, h, w});

    return x;
}

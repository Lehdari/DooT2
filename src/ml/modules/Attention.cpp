//
// Project: DooT2
// File: Attention.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/Attention.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


AttentionImpl::AttentionImpl(int dim, int heads, int headDim) :
    _heads          (heads),
    _innerDim       (headDim * heads),
    _scale          (std::pow((double)headDim, -0.5)),
    _ln1            (nn::LayerNormOptions({dim})),
    _linear1        (nn::LinearOptions(dim, _innerDim*3)),
    _ln2q           (nn::LayerNormOptions({headDim})),
    _ln2k           (nn::LayerNormOptions({headDim})),
    _softmax1       (nn::SoftmaxOptions(-1)),
    _linear2        (nn::LinearOptions(_innerDim, dim))
{
    register_module("ln1", _ln1);
    register_module("linear1", _linear1);
    register_module("ln2q", _ln2q);
    register_module("ln2k", _ln2k);
    register_module("softmax1", _softmax1);
    register_module("linear2", _linear2);
}

torch::Tensor AttentionImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    int b = x.sizes()[0];
    int s = x.sizes()[1];
    int n = x.sizes()[2];

    // Normalization
    x = _ln1(x);

    // Produce queries, keys and values
    x = _linear1(x);
    x = torch::permute(torch::reshape(x, {b, s, 3, _heads, -1}), {2, 0, 3, 1, 4}); // 3BHSN (H: heads)
    Tensor q = x.index({0});
    Tensor k = x.index({1});
    Tensor v = x.index({2});

    // Normalization
    x = _ln2q(x);
    x = _ln2k(x);

    // Dot product attention
    x = torch::matmul(q, k.transpose(-1, -2)) * _scale;
    x = _softmax1(x);

    // Gather attention-weighed output
    x = torch::matmul(x, v);

    // Output linear layer and final reshape
    x = torch::reshape(torch::permute(x, {0, 2, 1, 3}), {b, s, _innerDim}); // BSN
    x = _linear2(x);

    return x;
}

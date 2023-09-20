//
// Project: DooT2
// File: Attention.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

// Standard attention module (1D)
class AttentionImpl : public torch::nn::Module {
public:
    explicit AttentionImpl(
        int dim,
        int heads,
        int headDim
    );

    // excepts BSN tensor
    torch::Tensor forward(torch::Tensor x);

private:
    int                     _heads;
    int                     _innerDim;
    double                  _scale;

    torch::nn::LayerNorm    _ln1;
    torch::nn::Linear       _linear1; // input to qkv
    torch::nn::LayerNorm    _ln2q;
    torch::nn::LayerNorm    _ln2k;
    torch::nn::Softmax      _softmax1; // attention
    torch::nn::Linear       _linear2; // attention to output

    torch::Tensor           _posEmbedding;
};
TORCH_MODULE(Attention);

} // namespace ml

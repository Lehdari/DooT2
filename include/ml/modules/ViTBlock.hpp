//
// Project: DooT2
// File: ViTBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/modules/TransformerBlock.hpp"


namespace ml {

// Vision Transformer (ViT) block adapted from
// https://github.com/lucidrains/vit-pytorch/blob/8208c859a5474b2d93b429202833fcd9f395ec30/vit_pytorch/simple_vit.py#L37
class ViTBlockImpl : public torch::nn::Module {
public:
    explicit ViTBlockImpl(
        int dim,
        int heads,
        int headDim,
        int linearDim,
        bool usePositionalEmbedding = true
    );

    torch::Tensor forward(torch::Tensor x);

private:
    bool                    _usePositionalEmbedding;

    torch::nn::LayerNorm    _ln1;
    TransformerBlock        _transformer1;

    torch::Tensor           _posEmbedding;
};
TORCH_MODULE(ViTBlock);

} // namespace ml

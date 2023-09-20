//
// Project: DooT2
// File: TransformerBlock.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/modules/Attention.hpp"


namespace ml {

// Vision Transformer (Transformer) block adapted from
// https://github.com/lucidrains/vit-pytorch/blob/8208c859a5474b2d93b429202833fcd9f395ec30/vit_pytorch/simple_vit.py#L37
class TransformerBlockImpl : public torch::nn::Module {
public:
    explicit TransformerBlockImpl(
        int dim,
        int heads,
        int headDim,
        int linearDim
    );

    torch::Tensor forward(torch::Tensor x);

private:
    Attention               _attention1;
    torch::nn::LayerNorm    _ln1;
    torch::nn::Linear       _linear1;
    torch::nn::Linear       _linear2;

    torch::Tensor           _posEmbedding;
};
TORCH_MODULE(TransformerBlock);

} // namespace ml

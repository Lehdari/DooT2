//
// Project: DooT2
// File: Attention2D.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

// Vision Transformer (ViT) attention module, adapted from
// https://github.com/lucidrains/vit-pytorch/blob/8208c859a5474b2d93b429202833fcd9f395ec30/vit_pytorch/simple_vit.py#L37
class Attention2DImpl : public torch::nn::Module {
public:
    explicit Attention2DImpl(
        int dim,
        int heads,
        int headDim
    );

    torch::Tensor forward(torch::Tensor x);

private:
    int                     _heads;
    int                     _innerDim;
    double                  _scale;

    torch::nn::LayerNorm    _ln1;
    torch::nn::Linear       _linear1; // input to qkv
    torch::nn::Softmax      _softmax1; // attention
    torch::nn::Linear       _linear2; // attention to output

    torch::Tensor           _posEmbedding;
};
TORCH_MODULE(Attention2D);

} // namespace ml

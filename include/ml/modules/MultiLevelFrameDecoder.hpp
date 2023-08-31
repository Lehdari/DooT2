//
// Project: DooT2
// File: MultiLevelFrameDecoder.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>

#include "ml/MultiLevelImage.hpp"
#include "ml/modules/MultiLevelDecoderModule.hpp"
#include "ml/modules/ResNeXtModule.hpp"


namespace ml {

class MultiLevelFrameDecoderImpl : public torch::nn::Module {
public:

    MultiLevelFrameDecoderImpl();

    MultiLevelImage forward(torch::Tensor x, double level);

private:
    torch::nn::Linear           _linear1;
    torch::nn::BatchNorm1d      _bn1;
    torch::nn::PReLU            _pRelu1;
    torch::nn::Linear           _linear2;
    torch::nn::ConvTranspose2d  _convTranspose1a;
    torch::nn::ConvTranspose2d  _convTranspose1b;
    torch::nn::BatchNorm2d      _bn2a;
    torch::nn::BatchNorm2d      _bn2b;
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn3;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bn4;
    torch::nn::Conv2d           _convAux;
    torch::nn::BatchNorm2d      _bnAux;
    torch::nn::Conv2d           _conv_Y;
    torch::nn::Conv2d           _conv_UV;
    MultiLevelDecoderModule     _decoder1;
    MultiLevelDecoderModule     _decoder2;
    MultiLevelDecoderModule     _decoder3;
    MultiLevelDecoderModule     _decoder4;
    MultiLevelDecoderModule     _decoder5;
    MultiLevelDecoderModule     _decoder6;
    MultiLevelDecoderModule     _decoder7;
};
TORCH_MODULE(MultiLevelFrameDecoder);

} // namespace ml

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
#include "ml/modules/AdaptiveConvTranspose2d.hpp"
#include "ml/modules/AdaptiveResNetConvBlock.hpp"
#include "ml/modules/ResNetLinearBlock.hpp"


namespace ml {

class MultiLevelFrameDecoderImpl : public torch::nn::Module {
public:

    MultiLevelFrameDecoderImpl();

    MultiLevelImage forward(torch::Tensor x, double level);

private:
    ResNetLinearBlock           _resBlock1a;
    ResNetLinearBlock           _resBlock2a;
    ResNetLinearBlock           _resBlock3a;
    ResNetLinearBlock           _resBlock4a;
    ResNetLinearBlock           _resBlock1b;
    ResNetLinearBlock           _resBlock2b;
    ResNetLinearBlock           _resBlock3b;
    ResNetLinearBlock           _resBlock4b;
    torch::nn::BatchNorm1d      _bn1;
    AdaptiveConvTranspose2d     _convTranspose1a;
    AdaptiveConvTranspose2d     _convTranspose1b;
    torch::nn::BatchNorm2d      _bn2a;
    torch::nn::BatchNorm2d      _bn2b;
    AdaptiveResNetConvBlock     _resConvBlock1;
    AdaptiveResNetConvBlock     _resConvBlock2;
    torch::nn::Conv2d           _convAux;
    torch::nn::BatchNorm2d      _bnAux;
    torch::nn::Conv2d           _conv_Y;
    torch::nn::Conv2d           _conv_UV;
    AdaptiveResNetConvBlock     _resConvBlock3;
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

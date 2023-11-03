//
// Project: DooT2
// File: MultiLevelFrameEncoder.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/MultiLevelImage.hpp"
#include "ml/modules/MultiLevelEncoderModule.hpp"
#include "ml/modules/ResNetConvBlock.hpp"
#include "ml/modules/ResNetLinearBlock.hpp"

#include <torch/torch.h>


namespace ml {

class MultiLevelFrameEncoderImpl : public torch::nn::Module {
public:
    // useLinearResBlock: Add linear residual block in the end
    explicit MultiLevelFrameEncoderImpl(int featureMultiplier);

    // outputs encoding, mask
    std::tuple<torch::Tensor, torch::Tensor> forward(const MultiLevelImage& img); // single frame input [B * C * H * W]
    std::tuple<torch::Tensor, torch::Tensor> forwardSeq(const MultiLevelImage& seq, int t); // sequence input [L * B * C * H * W], t selects the time step

private:
    // Layers
    MultiLevelEncoderModule     _encoder1;
    MultiLevelEncoderModule     _encoder2;
    MultiLevelEncoderModule     _encoder3;
    MultiLevelEncoderModule     _encoder4;
    MultiLevelEncoderModule     _encoder5;
    MultiLevelEncoderModule     _encoder6;
    MultiLevelEncoderModule     _encoder7;
    torch::nn::BatchNorm2d      _bn1;
    torch::nn::Conv2d           _conv1;
    ResNetFourierConvBlock      _resFourierConvBlock1;
    ResNetFourierConvBlock      _resFourierConvBlock2;
    ResNetLinearBlock           _resBlock3a;
    ResNetLinearBlock           _resBlock3b;
    ResNetLinearBlock           _resBlock4;
    torch::nn::BatchNorm1d      _bn2a;
    torch::nn::BatchNorm1d      _bn2b;
    torch::nn::Linear           _linear1a;
    torch::nn::Linear           _linear1b;

    std::tuple<torch::Tensor, torch::Tensor> forwardCommon(torch::Tensor& x);
};
TORCH_MODULE(MultiLevelFrameEncoder);

} // namespace ml

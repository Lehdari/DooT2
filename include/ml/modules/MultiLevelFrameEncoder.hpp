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

#include <torch/torch.h>


namespace ml {

class MultiLevelFrameEncoderImpl : public torch::nn::Module {
public:
    // useLinearResBlock: Add linear residual block in the end
    MultiLevelFrameEncoderImpl(int featureMultiplier, bool useLinearResBlock);

    torch::Tensor forward(const MultiLevelImage& img);

private:
    bool                        _useLinearResBlock;
    MultiLevelEncoderModule     _encoder1;
    MultiLevelEncoderModule     _encoder2;
    MultiLevelEncoderModule     _encoder3;
    MultiLevelEncoderModule     _encoder4;
    MultiLevelEncoderModule     _encoder5;
    MultiLevelEncoderModule     _encoder6;
    MultiLevelEncoderModule     _encoder7;
    torch::nn::Conv2d           _conv1;
    torch::nn::BatchNorm2d      _bn1;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm1d      _bn2;
    torch::nn::Linear           _linear1;
    torch::nn::BatchNorm1d      _bn3;
    torch::nn::Linear           _linear2;
};
TORCH_MODULE(MultiLevelFrameEncoder);

} // namespace ml

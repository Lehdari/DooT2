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

#include "ml/modules/ResNeXtModule.hpp"


namespace ml {

class MultiLevelFrameDecoderImpl : public torch::nn::Module {
public:
    using ReturnType = std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>;

    MultiLevelFrameDecoderImpl();

    ReturnType forward(torch::Tensor x, double lossLevel);

private:
    torch::nn::Linear           _linear1;
    torch::nn::BatchNorm1d      _bn0_1;
    torch::nn::Linear           _linear2;
    torch::nn::BatchNorm1d      _bn0_2;
    ResNeXtModule               _resNext1a;
    ResNeXtModule               _resNext1b;
    torch::nn::ConvTranspose2d  _convTranspose1;
    torch::nn::BatchNorm2d      _bn1;
    ResNeXtModule               _resNext2a;
    ResNeXtModule               _resNext2b;
    torch::nn::ConvTranspose2d  _convTranspose2;
    torch::nn::BatchNorm2d      _bn2;
    torch::nn::Conv2d           _conv2;
    torch::nn::BatchNorm2d      _bn2b;
    torch::nn::Conv2d           _conv2_Y;
    torch::nn::Conv2d           _conv2_UV;
    ResNeXtModule               _resNext3a;
    ResNeXtModule               _resNext3b;
    torch::nn::ConvTranspose2d  _convTranspose3;
    torch::nn::BatchNorm2d      _bn3;
    torch::nn::Conv2d           _conv3;
    torch::nn::BatchNorm2d      _bn3b;
    torch::nn::Conv2d           _conv3_Y;
    torch::nn::Conv2d           _conv3_UV;
    ResNeXtModule               _resNext4a;
    ResNeXtModule               _resNext4b;
    torch::nn::ConvTranspose2d  _convTranspose4;
    torch::nn::BatchNorm2d      _bn4;
    torch::nn::Conv2d           _conv4;
    torch::nn::BatchNorm2d      _bn4b;
    torch::nn::Conv2d           _conv4_Y;
    torch::nn::Conv2d           _conv4_UV;
    ResNeXtModule               _resNext5a;
    ResNeXtModule               _resNext5b;
    torch::nn::ConvTranspose2d  _convTranspose5;
    torch::nn::BatchNorm2d      _bn5;
    torch::nn::Conv2d           _conv5;
    torch::nn::BatchNorm2d      _bn5b;
    torch::nn::Conv2d           _conv5_Y;
    torch::nn::Conv2d           _conv5_UV;
    ResNeXtModule               _resNext6a;
    ResNeXtModule               _resNext6b;
    torch::nn::ConvTranspose2d  _convTranspose6;
    torch::nn::BatchNorm2d      _bn6;
    torch::nn::Conv2d           _conv6;
    torch::nn::BatchNorm2d      _bn6b;
    torch::nn::Conv2d           _conv6_Y;
    torch::nn::Conv2d           _conv6_UV;
    ResNeXtModule               _resNext7a;
    ResNeXtModule               _resNext7b;
    torch::nn::ConvTranspose2d  _convTranspose7;
    torch::nn::BatchNorm2d      _bn7;
    torch::nn::Conv2d           _conv7;
    torch::nn::BatchNorm2d      _bn7b;
    torch::nn::Conv2d           _conv7_Y;
    torch::nn::Conv2d           _conv7_UV;
};
TORCH_MODULE(MultiLevelFrameDecoder);

} // namespace ml

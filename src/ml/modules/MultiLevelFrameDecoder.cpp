//
// Project: DooT2
// File: MultiLevelFrameDecoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelFrameDecoder.hpp"


using namespace ml;
using namespace torch;


inline torch::Tensor maxLoss(const torch::Tensor& target, const torch::Tensor& pred)
{
    return torch::max(torch::abs(target-pred));
}


MultiLevelFrameDecoderImpl::MultiLevelFrameDecoderImpl() :
    _resNext1a          (128, 256, 8, 64),
    _resNext1b          (256, 512, 8, 128),
    _convTranspose1     (nn::ConvTranspose2dOptions(512, 1024, {2, 2}).bias(false)),
    _bn1                (nn::BatchNorm2dOptions(1024)),
    _resNext2a          (1024, 1024, 8, 256),
    _resNext2b          (1024, 1024, 8, 256),
    _convTranspose2     (nn::ConvTranspose2dOptions(1024, 512, {5, 6}).stride({3, 4}).bias(false)),
    _bn2                (nn::BatchNorm2dOptions(512)),
    _conv2              (nn::Conv2dOptions(512, 16, {3, 3}).bias(false).padding(1)),
    _bn2b               (nn::BatchNorm2dOptions(16)),
    _conv2_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv2_UV           (nn::Conv2dOptions(16, 2, {1, 1})),
    _resNext3a          (512, 512, 8, 128),
    _resNext3b          (512, 512, 8, 128),
    _convTranspose3     (nn::ConvTranspose2dOptions(512, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bn3                (nn::BatchNorm2dOptions(256)),
    _conv3              (nn::Conv2dOptions(256, 16, {3, 3}).bias(false).padding(1)),
    _bn3b               (nn::BatchNorm2dOptions(16)),
    _conv3_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv3_UV           (nn::Conv2dOptions(16, 2, {1, 1})),
    _resNext4a          (256, 256, 8, 64),
    _resNext4b          (256, 256, 8, 64),
    _convTranspose4     (nn::ConvTranspose2dOptions(256, 128, {4, 4}).stride({2, 2}).bias(false)),
    _bn4                (nn::BatchNorm2dOptions(128)),
    _conv4              (nn::Conv2dOptions(128, 16, {3, 3}).bias(false).padding(1)),
    _bn4b               (nn::BatchNorm2dOptions(16)),
    _conv4_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv4_UV           (nn::Conv2dOptions(16, 2, {1, 1})),
    _resNext5a          (128, 128, 8, 32),
    _resNext5b          (128, 128, 8, 32),
    _convTranspose5     (nn::ConvTranspose2dOptions(128, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bn5                (nn::BatchNorm2dOptions(64)),
    _conv5              (nn::Conv2dOptions(64, 16, {3, 3}).bias(false).padding(1)),
    _bn5b               (nn::BatchNorm2dOptions(16)),
    _conv5_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv5_UV           (nn::Conv2dOptions(16, 2, {1, 1})),
    _resNext6a          (64, 64, 8, 16),
    _resNext6b          (64, 64, 8, 16),
    _convTranspose6     (nn::ConvTranspose2dOptions(64, 32, {4, 4}).stride({2, 2}).bias(false)),
    _bn6                (nn::BatchNorm2dOptions(32)),
    _conv6              (nn::Conv2dOptions(32, 16, {3, 3}).bias(false).padding(1)),
    _bn6b               (nn::BatchNorm2dOptions(16)),
    _conv6_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv6_UV           (nn::Conv2dOptions(16, 2, {1, 1})),
    _resNext7a          (32, 32, 8, 8),
    _resNext7b          (32, 32, 8, 8),
    _convTranspose7     (nn::ConvTranspose2dOptions(32, 32, {4, 4}).stride({2, 2}).bias(false)),
    _bn7                (nn::BatchNorm2dOptions(32)),
    _conv7              (nn::Conv2dOptions(32, 16, {1, 1}).bias(false)),
    _bn7b               (nn::BatchNorm2dOptions(16)),
    _conv7_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv7_UV           (nn::Conv2dOptions(16, 2, {1, 1}))
{
    register_module("resNext1a", _resNext1a);
    register_module("resNext1b", _resNext1b);
    register_module("convTranspose1", _convTranspose1);
    register_module("bn1", _bn1);
    register_module("resNext2a", _resNext2a);
    register_module("resNext2b", _resNext2b);
    register_module("convTranspose2", _convTranspose2);
    register_module("bn2", _bn2);
    register_module("conv2", _conv2);
    register_module("bn2b", _bn2b);
    register_module("conv2_Y", _conv2_Y);
    register_module("conv2_UV", _conv2_UV);
    register_module("resNext3a", _resNext3a);
    register_module("resNext3b", _resNext3b);
    register_module("convTranspose3", _convTranspose3);
    register_module("bn3", _bn3);
    register_module("conv3", _conv3);
    register_module("bn3b", _bn3b);
    register_module("conv3_Y", _conv3_Y);
    register_module("conv3_UV", _conv3_UV);
    register_module("resNext4a", _resNext4a);
    register_module("resNext4b", _resNext4b);
    register_module("convTranspose4", _convTranspose4);
    register_module("bn4", _bn4);
    register_module("conv4", _conv4);
    register_module("bn4b", _bn4b);
    register_module("conv4_Y", _conv4_Y);
    register_module("conv4_UV", _conv4_UV);
    register_module("resNext5a", _resNext5a);
    register_module("resNext5b", _resNext5b);
    register_module("convTranspose5", _convTranspose5);
    register_module("bn5", _bn5);
    register_module("conv5", _conv5);
    register_module("bn5b", _bn5b);
    register_module("conv5_Y", _conv5_Y);
    register_module("conv5_UV", _conv5_UV);
    register_module("resNext6a", _resNext6a);
    register_module("resNext6b", _resNext6b);
    register_module("convTranspose6", _convTranspose6);
    register_module("bn6", _bn6);
    register_module("conv6", _conv6);
    register_module("bn6b", _bn6b);
    register_module("conv6_Y", _conv6_Y);
    register_module("conv6_UV", _conv6_UV);
    register_module("resNext7a", _resNext7a);
    register_module("resNext7b", _resNext7b);
    register_module("convTranspose7", _convTranspose7);
    register_module("bn7", _bn7);
    register_module("conv7", _conv7);
    register_module("bn7b", _bn7b);
    register_module("conv7_Y", _conv7_Y);
    register_module("conv7_UV", _conv7_UV);

    auto* w = _conv2_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv2_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv3_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv3_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv4_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv4_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv5_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv5_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv6_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv6_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv7_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv7_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);

    w = _conv2_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv2_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv3_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv3_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv4_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv4_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv5_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv5_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv6_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv6_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv7_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv7_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

MultiLevelFrameDecoderImpl::ReturnType MultiLevelFrameDecoderImpl::forward(torch::Tensor x, double lossLevel)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    int batchSize = x.sizes()[0];
    auto device = x.device();

    // Decoder
    x = torch::reshape(x, {batchSize, 128, 4, 4});
    x = _resNext1a(x);
    x = torch::leaky_relu(_bn1(_convTranspose1(_resNext1b(x))), leakyReluNegativeSlope); // 5x5x1024

    x = torch::leaky_relu(_bn2(_convTranspose2(_resNext2b(_resNext2a(x)))), leakyReluNegativeSlope);
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x15x512

    torch::Tensor x0 = torch::leaky_relu(_bn2b(_conv2(x)), leakyReluNegativeSlope);
    torch::Tensor x0_Y = 0.5f + 0.51f * torch::tanh(_conv2_Y(x0));
    torch::Tensor x0_UV = 0.51f * torch::tanh(_conv2_UV(x0));
    x0 = torch::cat({x0_Y, x0_UV}, 1);

    torch::Tensor x1, x2, x3, x4, x5;
    if (lossLevel > 0.0) {
        x = torch::leaky_relu(_bn3(_convTranspose3(_resNext3b(_resNext3a(x)))), leakyReluNegativeSlope);
        x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x30x256

        x1 = torch::leaky_relu(_bn3b(_conv3(x)), leakyReluNegativeSlope);
        torch::Tensor x1_Y = 0.5f + 0.51f * torch::tanh(_conv3_Y(x1));
        torch::Tensor x1_UV = 0.51f * torch::tanh(_conv3_UV(x1));
        x1 = torch::cat({x1_Y, x1_UV}, 1);

        if (lossLevel > 1.0) {
            x = torch::leaky_relu(_bn4(_convTranspose4(_resNext4b(_resNext4a(x)))), leakyReluNegativeSlope);
            x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x60x128

            x2 = torch::leaky_relu(_bn4b(_conv4(x)), leakyReluNegativeSlope);
            torch::Tensor x2_Y = 0.5f + 0.51f * torch::tanh(_conv4_Y(x2));
            torch::Tensor x2_UV = 0.51f * torch::tanh(_conv4_UV(x2));
            x2 = torch::cat({x2_Y, x2_UV}, 1);

            if (lossLevel > 2.0) {
                x = torch::leaky_relu(_bn5(_convTranspose5(_resNext5b(_resNext5a(x)))), leakyReluNegativeSlope);
                x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 160x120x64

                x3 = torch::leaky_relu(_bn5b(_conv5(x)), leakyReluNegativeSlope);
                torch::Tensor x3_Y = 0.5f + 0.51f * torch::tanh(_conv5_Y(x3));
                torch::Tensor x3_UV = 0.51f * torch::tanh(_conv5_UV(x3));
                x3 = torch::cat({x3_Y, x3_UV}, 1);

                if (lossLevel > 3.0) {
                    x = torch::leaky_relu(_bn6(_convTranspose6(_resNext6b(_resNext6a(x)))), leakyReluNegativeSlope);
                    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x32

                    x4 = torch::leaky_relu(_bn6b(_conv6(x)), leakyReluNegativeSlope);
                    torch::Tensor x4_Y = 0.5f + 0.51f * torch::tanh(_conv6_Y(x4));
                    torch::Tensor x4_UV = 0.51f * torch::tanh(_conv6_UV(x4));
                    x4 = torch::cat({x4_Y, x4_UV}, 1);

                    if (lossLevel > 4.0) {
                        x = torch::leaky_relu(_bn7(_convTranspose7(_resNext7b(_resNext7a(x)))), leakyReluNegativeSlope);
                        x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 640x480x32

                        x5 = torch::leaky_relu(_bn7b(_conv7(x)), leakyReluNegativeSlope);
                        torch::Tensor x5_Y = 0.5f + 0.51f * torch::tanh(_conv7_Y(x5));
                        torch::Tensor x5_UV = 0.51f * torch::tanh(_conv7_UV(x5));
                        x5 = torch::cat({x5_Y, x5_UV}, 1);
                    }
                    else
                        x5 = torch::zeros({x.sizes()[0], 3, 480, 640}, TensorOptions().device(device));
                }
                else {
                    x4 = torch::zeros({x.sizes()[0], 3, 240, 320}, TensorOptions().device(device));
                    x5 = torch::zeros({x.sizes()[0], 3, 480, 640}, TensorOptions().device(device));
                }
            }
            else {
                x3 = torch::zeros({x.sizes()[0], 3, 120, 160}, TensorOptions().device(device));
                x4 = torch::zeros({x.sizes()[0], 3, 240, 320}, TensorOptions().device(device));
                x5 = torch::zeros({x.sizes()[0], 3, 480, 640}, TensorOptions().device(device));
            }
        }
        else {
            x2 = torch::zeros({x.sizes()[0], 3, 60, 80}, TensorOptions().device(device));
            x3 = torch::zeros({x.sizes()[0], 3, 120, 160}, TensorOptions().device(device));
            x4 = torch::zeros({x.sizes()[0], 3, 240, 320}, TensorOptions().device(device));
            x5 = torch::zeros({x.sizes()[0], 3, 480, 640}, TensorOptions().device(device));
        }
    }
    else {
        x1 = torch::zeros({x.sizes()[0], 3, 30, 40}, TensorOptions().device(device));
        x2 = torch::zeros({x.sizes()[0], 3, 60, 80}, TensorOptions().device(device));
        x3 = torch::zeros({x.sizes()[0], 3, 120, 160}, TensorOptions().device(device));
        x4 = torch::zeros({x.sizes()[0], 3, 240, 320}, TensorOptions().device(device));
        x5 = torch::zeros({x.sizes()[0], 3, 480, 640}, TensorOptions().device(device));
    }

    return {x0, x1, x2, x3, x4, x5};
}

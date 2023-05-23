//
// Project: DooT2
// File: MultiLevelFrameEncoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelFrameEncoder.hpp"


using namespace ml;
using namespace torch;


MultiLevelFrameEncoderImpl::MultiLevelFrameEncoderImpl() :
    _conv1          (nn::Conv2dOptions(3, 16, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn1            (nn::BatchNorm2dOptions(16)),
    _conv1b         (nn::Conv2dOptions(3, 16, {3, 3}).bias(false).padding(1)),
    _bn1b           (nn::BatchNorm2dOptions(16)),
    _conv2          (nn::Conv2dOptions(16, 32, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn2            (nn::BatchNorm2dOptions(32)),
    _conv2b         (nn::Conv2dOptions(3, 32, {3, 3}).bias(false).padding(1)),
    _bn2b           (nn::BatchNorm2dOptions(32)),
    _conv3          (nn::Conv2dOptions(32, 64, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn3            (nn::BatchNorm2dOptions(64)),
    _conv3b         (nn::Conv2dOptions(3, 64, {3, 3}).bias(false).padding(1)),
    _bn3b           (nn::BatchNorm2dOptions(64)),
    _conv4          (nn::Conv2dOptions(64, 128, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn4            (nn::BatchNorm2dOptions(128)),
    _conv4b         (nn::Conv2dOptions(3, 128, {3, 3}).bias(false).padding(1)),
    _bn4b           (nn::BatchNorm2dOptions(128)),
    _conv5          (nn::Conv2dOptions(128, 256, {4, 4}).stride({2, 2}).bias(false).padding(1)),
    _bn5            (nn::BatchNorm2dOptions(256)),
    _conv5b         (nn::Conv2dOptions(3, 256, {3, 3}).bias(false).padding(1)),
    _bn5b           (nn::BatchNorm2dOptions(256)),
    _conv6          (nn::Conv2dOptions(256, 512, {5, 6}).stride({3, 4}).bias(false).padding(1)),
    _bn6            (nn::BatchNorm2dOptions(512)),
    _conv7          (nn::Conv2dOptions(512, 512, {2, 2}).bias(false)),
    _bn7            (nn::BatchNorm2dOptions(512)),
    _conv8          (nn::Conv2dOptions(512, 128, {1, 1}).bias(false)),
    _bn8            (nn::BatchNorm1dOptions(2048)),
    _linear1        (nn::LinearOptions(2048, 1024)),
    _bn9            (nn::BatchNorm1dOptions(1024)),
    _linear2        (nn::LinearOptions(1024, 2048))
{
    register_module("conv1", _conv1);
    register_module("bn1", _bn1);
    register_module("conv1b", _conv1b);
    register_module("bn1b", _bn1b);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    register_module("conv2b", _conv2b);
    register_module("bn2b", _bn2b);
    register_module("conv3", _conv3);
    register_module("bn3", _bn3);
    register_module("conv3b", _conv3b);
    register_module("bn3b", _bn3b);
    register_module("conv4", _conv4);
    register_module("bn4", _bn4);
    register_module("conv4b", _conv4b);
    register_module("bn4b", _bn4b);
    register_module("conv5", _conv5);
    register_module("bn5", _bn5);
    register_module("conv5b", _conv5b);
    register_module("bn5b", _bn5b);
    register_module("conv6", _conv6);
    register_module("bn6", _bn6);
    register_module("conv7", _conv7);
    register_module("bn7", _bn7);
    register_module("conv8", _conv8);
    register_module("bn8", _bn8);
    register_module("linear1", _linear1);
    register_module("bn9", _bn9);
    register_module("linear2", _linear2);

    auto* w = _conv8->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.0001);
    w = _linear2->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.0001);
    w = _linear2->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

torch::Tensor MultiLevelFrameEncoderImpl::forward(
    torch::Tensor x5,
    torch::Tensor x4,
    torch::Tensor x3,
    torch::Tensor x2,
    torch::Tensor x1,
    torch::Tensor x0,
    double lossLevel
) {
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    // Encoder
    torch::Tensor x;
    if (lossLevel > 0.0) {
        if (lossLevel > 1.0) {
            if (lossLevel > 2.0) {
                if (lossLevel > 3.0) {
                    if (lossLevel > 4.0) {
                        x = torch::leaky_relu(_bn1(_conv1(x5)), leakyReluNegativeSlope); // 320x240x16
                        x4 = torch::leaky_relu(_bn1b(_conv1b(x4)), leakyReluNegativeSlope);
                        float w4 = (float)std::clamp(5.0-lossLevel, 0.0, 1.0);
                        x = w4*x4 + (1.0f-w4)*x;
                    }
                    else
                        x = torch::leaky_relu(_bn1b(_conv1b(x4)), leakyReluNegativeSlope);

                    x = torch::leaky_relu(_bn2(_conv2(x)), leakyReluNegativeSlope); // 160x120x32
                    x3 = torch::leaky_relu(_bn2b(_conv2b(x3)), leakyReluNegativeSlope);
                    float w3 = (float)std::clamp(4.0-lossLevel, 0.0, 1.0);
                    x = w3*x3 + (1.0f-w3)*x;
                }
                else
                    x = torch::leaky_relu(_bn2b(_conv2b(x3)), leakyReluNegativeSlope);

                x = torch::leaky_relu(_bn3(_conv3(x)), leakyReluNegativeSlope); // 80x60x64
                x2 = torch::leaky_relu(_bn3b(_conv3b(x2)), leakyReluNegativeSlope);
                float w2 = (float)std::clamp(3.0-lossLevel, 0.0, 1.0);
                x = w2*x2 + (1.0f-w2)*x;
            }
            else
                x = torch::leaky_relu(_bn3b(_conv3b(x2)), leakyReluNegativeSlope);

            x = torch::leaky_relu(_bn4(_conv4(x)), leakyReluNegativeSlope); // 40x30x128
            x1 = torch::leaky_relu(_bn4b(_conv4b(x1)), leakyReluNegativeSlope);
            float w1 = (float)std::clamp(2.0-lossLevel, 0.0, 1.0);
            x = w1*x1 + (1.0f-w1)*x;
        }
        else
            x = torch::leaky_relu(_bn4b(_conv4b(x1)), leakyReluNegativeSlope);

        x = torch::leaky_relu(_bn5(_conv5(x)), leakyReluNegativeSlope); // 20x15x256
        x0 = torch::leaky_relu(_bn5b(_conv5b(x0)), leakyReluNegativeSlope);
        float w0 = (float)std::clamp(1.0-lossLevel, 0.0, 1.0);
        x = w0*x0 + (1.0f-w0)*x;
    }
    else
        x = torch::leaky_relu(_bn5b(_conv5b(x0)), leakyReluNegativeSlope);

    x = torch::leaky_relu(_bn6(_conv6(x)), leakyReluNegativeSlope); // 5x5x512
    x = torch::leaky_relu(_bn7(_conv7(x)), leakyReluNegativeSlope); // 4x4x512
    x = _conv8(x); // 4x4x128
    x = torch::reshape(x, {-1, 2048}); // 2048

    // linear residual block
    torch::Tensor y = torch::leaky_relu(_bn8(x), leakyReluNegativeSlope);
    x = x + _linear2(torch::leaky_relu(_bn9(_linear1(y)), leakyReluNegativeSlope));

    return x;
}

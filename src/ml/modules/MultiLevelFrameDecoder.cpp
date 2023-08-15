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
using namespace torch::indexing;
namespace tf = torch::nn::functional;


MultiLevelFrameDecoderImpl::MultiLevelFrameDecoderImpl() :
    _linear1            (nn::LinearOptions(2048, 2048).bias(false)),
    _bn1                (nn::BatchNorm1dOptions(2048)),
    _pRelu1             (nn::PReLUOptions().num_parameters(2048).init(0.01)),
    _convTranspose1a    (nn::ConvTranspose2dOptions(128, 128, {2, 2}).bias(false)),
    _convTranspose1b    (nn::ConvTranspose2dOptions(2048, 128, {5, 5}).bias(false).groups(8)),
    _bn2a               (nn::BatchNorm2dOptions(128)),
    _bn2b               (nn::BatchNorm2dOptions(128)),
    _convAux            (nn::Conv2dOptions(256, 16, {3, 3}).bias(false).padding(1)),
    _bnAux              (nn::BatchNorm2dOptions(16)),
    _conv_Y             (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv_UV            (nn::Conv2dOptions(16, 2, {1, 1})),
    _decoder1           (0.0, 256, 512, 512, 8, 16, 64, 64, 10, 15, 3, 2),
    _decoder2           (1.0, 512, 512, 512, 16, 16, 64, 64, 20, 15, 1, 2),
    _decoder3           (2.0, 512, 512, 256, 16, 16, 64, 64, 40, 30, 2, 2),
    _decoder4           (3.0, 256, 256, 128, 8, 8, 64, 64, 80, 60, 2, 2),
    _decoder5           (4.0, 128, 128, 64, 8, 8, 32, 32, 160, 120, 2, 2),
    _decoder6           (5.0, 64, 64, 32, 8, 8, 16, 16, 320, 240, 2, 2),
    _decoder7           (6.0, 32, 32, 32, 8, 8, 8, 8, 640, 480, 2, 2)
{
    register_module("linear0_1", _linear1);
    register_module("bn1", _bn1);
    register_module("pRelu1", _pRelu1);
    register_module("convTranspose1a", _convTranspose1a);
    register_module("convTranspose1b", _convTranspose1b);
    register_module("bn2a", _bn2a);
    register_module("bn2b", _bn2b);
    register_module("convAux", _convAux);
    register_module("bnAux", _bnAux);
    register_module("conv_Y", _conv_Y);
    register_module("conv_UV", _conv_UV);
    register_module("decoder1", _decoder1);
    register_module("decoder2", _decoder2);
    register_module("decoder3", _decoder3);
    register_module("decoder4", _decoder4);
    register_module("decoder5", _decoder5);
    register_module("decoder6", _decoder6);
    register_module("decoder7", _decoder7);

    auto* w = _conv_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _conv_Y->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
    w = _conv_UV->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

MultiLevelImage MultiLevelFrameDecoderImpl::forward(torch::Tensor x, double level)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    int batchSize = x.sizes()[0];
#if 0
    // Use dropout directly on the encodings with rate decreasing according to the level
    if (this->is_training()) {
        constexpr double passBase = 0.8;
        constexpr double passLinear = 0.1;
        constexpr double passExponential = 0.1;
        double dropoutRate = 1.0-(passBase + passLinear*(level/7.0) + passExponential*(std::pow(2.0, level)/128.0));
        x = tf::dropout(x, tf::DropoutFuncOptions().p(dropoutRate));
    }
#endif
    // Decoder
    // Linear residual layer
    x = x + _bn1(_pRelu1(_linear1(x)));

    // 2-way deconv into 5x5x256
    torch::Tensor y = torch::reshape(x, {batchSize, 128, 4, 4});
    y = _bn2a(_convTranspose1a(y));
    x = torch::reshape(x, {batchSize, 2048, 1, 1});
    x = _bn2b(_convTranspose1b(x));
    x = torch::leaky_relu(torch::cat({x, y}, 1), leakyReluNegativeSlope);

    MultiLevelImage img;
    img.level = level;

    // 5x5 auxiliary image output
    y = torch::leaky_relu(_bnAux(_convAux(x)), leakyReluNegativeSlope);
    torch::Tensor y_Y = 0.5f + 0.51f * torch::tanh(_conv_Y(y));
    torch::Tensor y_UV = 0.51f * torch::tanh(_conv_UV(y));
    img.img0 = torch::cat({y_Y, y_UV}, 1);

    // Rest of the decoder layers
    std::tie(x, img.img1) = _decoder1(x, level); // 10x15
    std::tie(x, img.img2) = _decoder2(x, level); // 20x15
    std::tie(x, img.img3) = _decoder3(x, level); // 40x30
    std::tie(x, img.img4) = _decoder4(x, level); // 80x60
    std::tie(x, img.img5) = _decoder5(x, level); // 160x120
    std::tie(x, img.img6) = _decoder6(x, level); // 320x240
    std::tie(x, img.img7) = _decoder7(x, level); // 640x480

    return img;
}

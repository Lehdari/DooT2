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
    _resBlock1          (2048, 2048, 2048),
    _convTranspose1a    (nn::ConvTranspose2dOptions(128, 128, {2, 2})),
    _convTranspose1b    (nn::ConvTranspose2dOptions(2048, 128, {5, 5}).groups(8)),
    _bn1a               (nn::BatchNorm2dOptions(128)),
    _bn1b               (nn::BatchNorm2dOptions(128)),
    _resBlock2          (256, 512, 512),
    _resBlock3          (512, 512, 512),
    _convAux            (nn::Conv2dOptions(512, 8, {1, 1}).bias(false)),
    _bnAux              (nn::BatchNorm2dOptions(8)),
    _conv_Y             (nn::Conv2dOptions(8, 1, {1, 1})),
    _conv_UV            (nn::Conv2dOptions(8, 2, {1, 1})),
    _decoder1           (0.0, 512, 512, 2, 3),
    _decoder2           (1.0, 512, 256, 2, 1),
    _decoder3           (2.0, 256, 128, 2, 2),
    _decoder4           (3.0, 128, 64, 2, 2),
    _decoder5           (4.0, 64, 32, 2, 2),
    _decoder6           (5.0, 32, 16, 2, 2),
    _decoder7           (6.0, 16, 8, 2, 2)
{
    register_module("resBlock1", _resBlock1);
    register_module("convTranspose1a", _convTranspose1a);
    register_module("convTranspose1b", _convTranspose1b);
    register_module("bn1a", _bn1a);
    register_module("bn1b", _bn1b);
    register_module("resBlock2", _resBlock2);
    register_module("resBlock3", _resBlock3);
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
    // Linear residual module
    x = _resBlock1(x);

    // 2-way deconv into 5x5x256
    torch::Tensor y = torch::reshape(x, {batchSize, 128, 4, 4});
    y = _bn1a(_convTranspose1a(y));
    x = torch::reshape(x, {batchSize, 2048, 1, 1});
    x = _bn1b(_convTranspose1b(x));
    x = torch::leaky_relu(torch::cat({x, y}, 1), leakyReluNegativeSlope);

    // Residual conv blocks
    x = _resBlock3(_resBlock2(x));

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

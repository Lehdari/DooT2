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
    _resBlock1a         (2048, 2048, 2048),
    _resBlock1b         (2048, 1024, 1024),
    _bn1                (nn::BatchNorm1dOptions(2048)),
    _convTranspose1a    (128, 128, 1024, std::vector<long>{2, 2}, 1, 16),
    _convTranspose1b    (2048, 128, 1024, std::vector<long>{5, 5}, 64, 8),
    _bn2a               (nn::BatchNorm2dOptions(128)),
    _bn2b               (nn::BatchNorm2dOptions(128)),
    _resBlock2          (256, 1024, 512, 1024, 64, 4, true, 0.0, true),
    _resBlock3          (512, 1024, 512, 1024, 64, 4, true, 0.0, true),
    _convAux            (nn::Conv2dOptions(512, 8, {1, 1}).bias(false)),
    _bnAux              (nn::BatchNorm2dOptions(8)),
    _conv_Y             (nn::Conv2dOptions(8, 1, {1, 1})),
    _conv_UV            (nn::Conv2dOptions(8, 2, {1, 1})),
    _resBlock4          (512, 1024, 512, 1024, 64, 4, true, 0.0, true),
    _decoder1           (0.0, 512, 512, 1024, 2, 3, 32, 64, 2, 4),
    _decoder2           (1.0, 512, 256, 1024, 2, 1, 16, 32, 2, 4),
    _decoder3           (2.0, 256, 128, 1024, 2, 2, 8, 16, 4, 8),
    _decoder4           (3.0, 128, 64, 1024, 2, 2, 4, 8, 4, 8),
    _decoder5           (4.0, 64, 32, 1024, 2, 2, 2, 4, 4, 16),
    _decoder6           (5.0, 32, 16, 1024, 2, 2, 1, 2, 4, 16),
    _decoder7           (6.0, 16, 8, 1024, 2, 2, 1, 1, 4, 16)
{
    register_module("resBlock1a", _resBlock1a);
    register_module("resBlock1b ", _resBlock1b );
    register_module("bn1", _bn1);
    register_module("convTranspose1a", _convTranspose1a);
    register_module("convTranspose1b", _convTranspose1b);
    register_module("bn2a", _bn2a);
    register_module("bn2b", _bn2b);
    register_module("resBlock2", _resBlock2);
    register_module("resBlock3", _resBlock3);
    register_module("convAux", _convAux);
    register_module("bnAux", _bnAux);
    register_module("conv_Y", _conv_Y);
    register_module("conv_UV", _conv_UV);
    register_module("resBlock4", _resBlock4);
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

    // Context directly fed into all decoder layers
    torch::Tensor context = _resBlock1b(x);

    // Main branch
    x = _resBlock1a(x);

    // 2-way deconv into 5x5x256
    x = gelu(_bn1(x), "tanh");
    torch::Tensor y = torch::reshape(x, {batchSize, 128, 4, 4});
    y = _bn2a(_convTranspose1a(y, context));
    x = torch::reshape(x, {batchSize, 2048, 1, 1});
    x = _bn2b(_convTranspose1b(x, context));
    x = gelu(torch::cat({x, y}, 1), "tanh");

    // First residual conv blocks
    x = _resBlock2(x, context);
    x = _resBlock3(x, context);

    // 5x5 auxiliary image output
    MultiLevelImage img;
    img.level = level;
    y = gelu(_bnAux(_convAux(x)), "tanh");
    torch::Tensor y_Y = 0.5f + 0.51f * torch::tanh(_conv_Y(y));
    torch::Tensor y_UV = 0.51f * torch::tanh(_conv_UV(y));
    img.img0 = torch::cat({y_Y, y_UV}, 1);

    // Second residual conv block
    x = _resBlock4(x, context);

    // Rest of the decoder layers
    std::tie(x, img.img1) = _decoder1(x, context, level, &img.img0); // 10x15
    std::tie(x, img.img2) = _decoder2(x, context, level, &img.img1); // 20x15
    std::tie(x, img.img3) = _decoder3(x, context, level, &img.img2); // 40x30
    std::tie(x, img.img4) = _decoder4(x, context, level, &img.img3); // 80x60
    std::tie(x, img.img5) = _decoder5(x, context, level, &img.img4); // 160x120
    std::tie(x, img.img6) = _decoder6(x, context, level, &img.img5); // 320x240
    std::tie(x, img.img7) = _decoder7(x, context, level, &img.img6); // 640x480

    return img;
}

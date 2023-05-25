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
    _encoder1       (6.0, 3, 16, ExpandingArray<2>{4,4}, ExpandingArray<2>{2,2}), // 320x240
    _encoder2       (5.0, 16, 32, ExpandingArray<2>{4,4}, ExpandingArray<2>{2,2}), // 160x120
    _encoder3       (4.0, 32, 64, ExpandingArray<2>{4,4}, ExpandingArray<2>{2,2}), // 80x60
    _encoder4       (3.0, 64, 128, ExpandingArray<2>{4,4}, ExpandingArray<2>{2,2}), // 40x30
    _encoder5       (2.0, 128, 256, ExpandingArray<2>{4,4}, ExpandingArray<2>{2,2}), // 20x15
    _encoder6       (1.0, 256, 512, ExpandingArray<2>{3,4}, ExpandingArray<2>{1,2}), // 10x15
    _encoder7       (0.0, 512, 1024, ExpandingArray<2>{5,4}, ExpandingArray<2>{3,2}), // 5x5
    _conv1          (nn::Conv2dOptions(1024, 512, {2, 2}).bias(false)),
    _bn1            (nn::BatchNorm2dOptions(512)),
    _conv2          (nn::Conv2dOptions(512, 128, {1, 1}).bias(false)),
    _bn2            (nn::BatchNorm1dOptions(2048)),
    _linear1        (nn::LinearOptions(2048, 1024)),
    _bn3            (nn::BatchNorm1dOptions(1024)),
    _linear2        (nn::LinearOptions(1024, 2048))
{
    register_module("encoder1", _encoder1);
    register_module("encoder2", _encoder2);
    register_module("encoder3", _encoder3);
    register_module("encoder4", _encoder4);
    register_module("encoder5", _encoder5);
    register_module("encoder6", _encoder6);
    register_module("encoder7", _encoder7);
    register_module("conv1", _conv1);
    register_module("bn1", _bn1);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    register_module("linear1", _linear1);
    register_module("bn3", _bn3);
    register_module("linear2", _linear2);

    auto* w = _conv2->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.0001);
    w = _linear2->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.0001);
    w = _linear2->named_parameters(false).find("bias");
    if (w == nullptr) throw std::runtime_error("Unable to find layer biases");
    torch::nn::init::zeros_(*w);
}

torch::Tensor MultiLevelFrameEncoderImpl::forward(const MultiLevelImage& img)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x = _encoder1(img.img7, img.img6, img.level); // 320x240x16
    x = _encoder2(x, img.img5, img.level); // 160x120x32
    x = _encoder3(x, img.img4, img.level); // 80x60x64
    x = _encoder4(x, img.img3, img.level); // 40x30x128
    x = _encoder5(x, img.img2, img.level); // 20x15x256
    x = _encoder6(x, img.img1, img.level); // 10x15x512
    x = _encoder7(x, img.img0, img.level); // 5x5x1024

    x = torch::leaky_relu(_bn1(_conv1(x)), leakyReluNegativeSlope); // 4x4x512
    x = _conv2(x); // 4x4x128
    x = torch::reshape(x, {-1, 2048}); // 2048

    // linear residual block
    torch::Tensor y = torch::leaky_relu(_bn2(x), leakyReluNegativeSlope);
    x = x + _linear2(torch::leaky_relu(_bn3(_linear1(y)), leakyReluNegativeSlope));

    return x;
}

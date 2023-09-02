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


MultiLevelFrameEncoderImpl::MultiLevelFrameEncoderImpl(int featureMultiplier) :
    _encoder1   (6.0, 3, 8, 2, 2), // 320x240
    _encoder2   (5.0, 8, 16, 2, 2), // 160x120
    _encoder3   (4.0, 16, 32, 2, 2), // 80x60
    _encoder4   (3.0, 32, 64, 2, 2), // 40x30
    _encoder5   (2.0, 64, 128, 2, 2), // 20x15
    _encoder6   (1.0, 128, 256, 2, 1), // 10x15
    _encoder7   (0.0, 256, 512, 2, 3), // 5x5
    _bn1        (nn::BatchNorm2dOptions(512)),
    _pRelu1     (nn::PReLUOptions().num_parameters(512).init(0.01)),
    _conv1      (nn::Conv2dOptions(512, 512, {2, 2}).bias(false).groups(4)),
    _resBlock1  (512, 256, 128, 0.01, 0.001),
    _resBlock2a (2048, 2048, 2048, 0.01, 0.001),
    _resBlock2b (2048, 2048, 2048, 0.01, 0.001),
    _bn2a       (nn::BatchNorm1dOptions(2048)),
    _bn2b       (nn::BatchNorm1dOptions(2048)),
    _linear1a   (nn::LinearOptions(2048, 2048).bias(false)),
    _linear1b   (nn::LinearOptions(2048, 2048).bias(false)),
    _maskGradientScale  (torch::ones({}))
{
    register_module("encoder1", _encoder1);
    register_module("encoder2", _encoder2);
    register_module("encoder3", _encoder3);
    register_module("encoder4", _encoder4);
    register_module("encoder5", _encoder5);
    register_module("encoder6", _encoder6);
    register_module("encoder7", _encoder7);
    register_module("bn1", _bn1);
    register_module("pRelu1", _pRelu1);
    register_module("conv1", _conv1);
    register_module("resBlock1", _resBlock1);
    register_module("resBlock2a", _resBlock2a);
    register_module("resBlock2b", _resBlock2b);
    register_module("bn2a", _bn2a);
    register_module("bn2b", _bn2b);
    register_module("linear1a", _linear1a);
    register_module("linear1b", _linear1b);

    auto* w = _linear1a->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
    w = _linear1b->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);

    _maskGradientScale = register_parameter("maskGradientScale", torch::ones({}), false);
}

std::tuple<torch::Tensor, torch::Tensor> MultiLevelFrameEncoderImpl::forward(const MultiLevelImage& img)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor x = _encoder1(img.img7, img.img6, img.level); // 320x240
    x = _encoder2(x, img.img5, img.level); // 160x120
    x = _encoder3(x, img.img4, img.level); // 80x60
    x = _encoder4(x, img.img3, img.level); // 40x30
    x = _encoder5(x, img.img2, img.level); // 20x15
    x = _encoder6(x, img.img1, img.level); // 10x15
    x = _encoder7(x, img.img0, img.level); // 5x5

    x = _resBlock1(_conv1(_pRelu1(_bn1(x))));
    x = torch::reshape(x, {-1, 2048}); // 2048

    // mask probabilities
    torch::Tensor maskProb = 0.5+0.5*torch::tanh(_linear1b(_bn2b(_resBlock2b(x))));

    // Final residual linear module on values
    x = _linear1a(_bn2a(_resBlock2a(x)));

    // Update the mask straight-through gradient scale
    constexpr double maskGradientRelativeScale = 0.01;
    double m = 0.99*_maskGradientScale.item<double>() +
        0.01*(maskGradientRelativeScale / x.abs().mean().item<double>());
    _maskGradientScale = torch::ones({})*m;

    torch::Tensor mask;
    if (this->is_training()) {
        torch::Tensor s = torch::rand(maskProb.sizes(),
            TensorOptions().device(maskProb.device()).dtype(maskProb.dtype()));
        mask = torch::where(maskProb < s, torch::zeros_like(maskProb), torch::ones_like(maskProb))
            + (maskProb - maskProb.detach())*_maskGradientScale; // straight-through gradient
    }
    else {
        mask = maskProb; // deterministic during inference
    }


    return { x*mask, maskProb };
}

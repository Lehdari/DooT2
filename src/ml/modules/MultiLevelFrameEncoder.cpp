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
    _encoder1           (5.5, 3, 8, 2, 2, 1, 4), // 320x240
    _encoder2           (4.5, 8, 16, 2, 2, 2, 4), // 160x120
    _encoder3           (3.5, 16, 32, 2, 2, 4, 4), // 80x60
    _encoder4           (2.5, 32, 64, 2, 2, 8, 4), // 40x30
    _encoder5           (1.5, 64, 128, 2, 2, 16, 4), // 20x15
    _encoder6           (0.5, 128, 256, 2, 1, 32, 4), // 10x15
    _encoder7           (-0.5, 256, 512, 2, 3, 64, 2), // 5x5
    _bn1                (nn::BatchNorm2dOptions(512)),
    _conv1              (nn::Conv2dOptions(512, 512, {2, 2}).bias(false).groups(512)),
    _resBlock1          (512, 1024, 256, 64, true, 0.001),
    _resBlock2          (256, 1024, 128, 64, true, 0.001),
    _resBlock3a         (2048, 1024, 2048, 0.001),
    _resBlock3b         (2048, 1024, 2048),
    _resBlock4          (2048, 2048, 2048),
    _bn2a               (nn::BatchNorm1dOptions(2048)),
    _bn2b               (nn::BatchNorm1dOptions(2048)),
    _linear1a           (nn::LinearOptions(2048, 2048)),
    _linear1b           (nn::LinearOptions(2048, 2048))
{
    register_module("encoder1", _encoder1);
    register_module("encoder2", _encoder2);
    register_module("encoder3", _encoder3);
    register_module("encoder4", _encoder4);
    register_module("encoder5", _encoder5);
    register_module("encoder6", _encoder6);
    register_module("encoder7", _encoder7);
    register_module("bn1", _bn1);
    register_module("conv1", _conv1);
    register_module("resBlock1", _resBlock1);
    register_module("resBlock2", _resBlock2);
    register_module("resBlock3a", _resBlock3a);
    register_module("resBlock3b", _resBlock3b);
    register_module("resBlock4", _resBlock4);
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
}

std::tuple<torch::Tensor, torch::Tensor> MultiLevelFrameEncoderImpl::forward(const MultiLevelImage& img)
{
    int b = img.img7.sizes()[0];

//    torch::Tensor x = _encoder1(img.img7, img.img6, img.level); // 320x240
//    x = _encoder2(x, img.img5, img.level); // 160x120
    torch::Tensor x = _encoder2(img.img6, img.img5, img.level); // 160x120
    x = _encoder3(x, img.img4, img.level); // 80x60
    x = _encoder4(x, img.img3, img.level); // 40x30
    x = _encoder5(x, img.img2, img.level); // 20x15
    x = _encoder6(x, img.img1, img.level); // 10x15
    x = _encoder7(x, img.img0, img.level); // 5x5

    x = _conv1(gelu(_bn1(x), "tanh"));
    x = _resBlock1(x);
    x = _resBlock2(x);
    x = torch::reshape(x, {b, x.sizes()[1]*x.sizes()[2]*x.sizes()[3]}); // 2048

    // mask probabilities
    torch::Tensor maskProb = x;
    maskProb = _resBlock3b(maskProb);
    maskProb = torch::reshape(maskProb, {b, 2048});
    maskProb = 0.5+0.5*torch::tanh(_linear1b(_bn2b(maskProb)));

    // Values branch
    x = _resBlock4(_resBlock3a(x));
    x = torch::reshape(x, {b, 2048});
    x = _linear1a(_bn2a(x));

    // Update the mask straight-through gradient scale
    constexpr double maskGradientRelativeScale = 0.01;
    torch::Tensor mask;
    if (this->is_training()) {
        torch::Tensor s = torch::rand(maskProb.sizes(),
            TensorOptions().device(maskProb.device()).dtype(maskProb.dtype()));
        mask = torch::where(maskProb < s, torch::zeros_like(maskProb), torch::ones_like(maskProb))
            + (maskProb - maskProb.detach())*maskGradientRelativeScale; // straight-through gradient
    }
    else {
        mask = maskProb; // deterministic during inference
    }

    return { x*mask, maskProb };
}

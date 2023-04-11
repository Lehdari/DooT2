//
// Project: DooT2
// File: FrameDecoder2.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FrameDecoder2.hpp"


using namespace torch;


inline torch::Tensor maxLoss(const torch::Tensor& target, const torch::Tensor& pred)
{
    return torch::max(torch::abs(target-pred));
}


FrameDecoder2Impl::FrameDecoder2Impl() :
    _resNext1a          (128, 256, 8, 64),
    _resNext1b          (256, 512, 8, 128),
    _convTranspose1     (nn::ConvTranspose2dOptions(512, 512, {2, 2}).bias(false)),
    _bn1                (nn::BatchNorm2dOptions(512)),
    _resNext2a          (512, 512, 8, 128),
    _resNext2b          (512, 512, 8, 128),
    _convTranspose2     (nn::ConvTranspose2dOptions(512, 256, {5, 6}).stride({3, 4}).bias(false)),
    _bn2                (nn::BatchNorm2dOptions(256)),
    _resNext3a          (256, 256, 8, 64),
    _resNext3b          (256, 256, 8, 64),
    _convTranspose3     (nn::ConvTranspose2dOptions(256, 256, {4, 4}).stride({2, 2}).bias(false)),
    _bn3                (nn::BatchNorm2dOptions(256)),
    _dropout3           (nn::DropoutOptions(0.25)),
    _resNext4a          (256, 256, 8, 64),
    _resNext4b          (256, 256, 8, 64),
    _convTranspose4     (nn::ConvTranspose2dOptions(256, 128, {4, 4}).stride({2, 2}).bias(false)),
    _bn4                (nn::BatchNorm2dOptions(128)),
    _dropout4           (nn::DropoutOptions(0.5)),
    _resNext5a          (128, 128, 8, 32),
    _resNext5b          (128, 128, 8, 32),
    _convTranspose5     (nn::ConvTranspose2dOptions(128, 64, {4, 4}).stride({2, 2}).bias(false)),
    _bn5                (nn::BatchNorm2dOptions(64)),
    _dropout5           (nn::DropoutOptions(0.625)),
    _resNext6a          (64, 64, 8, 16),
    _resNext6b          (64, 64, 8, 16),
    _convTranspose6     (nn::ConvTranspose2dOptions(64, 32, {4, 4}).stride({2, 2}).bias(false)),
    _bn6                (nn::BatchNorm2dOptions(32)),
    _dropout6           (nn::DropoutOptions(0.75)),
    _resNext7a          (32, 32, 8, 8),
    _resNext7b          (32, 32, 8, 8),
    _convTranspose7     (nn::ConvTranspose2dOptions(32, 32, {4, 4}).stride({2, 2}).bias(false)),
    _bn7                (nn::BatchNorm2dOptions(32)),
    _conv8              (nn::Conv2dOptions(32, 16, {3, 3}).bias(false).padding(1)),
    _bn8                (nn::BatchNorm2dOptions(16)),
    _conv8_Y            (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv8_UV           (nn::Conv2dOptions(16, 2, {1, 1}))
{
    register_module("resNext1a", _resNext1a);
    register_module("resNext1b", _resNext1b);
    register_module("convTranspose1", _convTranspose1);
    register_module("bn1", _bn1);
    register_module("resNext2a", _resNext2a);
    register_module("resNext2b", _resNext2b);
    register_module("convTranspose2", _convTranspose2);
    register_module("bn2", _bn2);
    register_module("resNext3a", _resNext3a);
    register_module("resNext3b", _resNext3b);
    register_module("convTranspose3", _convTranspose3);
    register_module("bn3", _bn3);
    register_module("dropout3", _dropout3);
    register_module("resNext4a", _resNext4a);
    register_module("resNext4b", _resNext4b);
    register_module("convTranspose4", _convTranspose4);
    register_module("bn4", _bn4);
    register_module("dropout4", _dropout4);
    register_module("resNext5a", _resNext5a);
    register_module("resNext5b", _resNext5b);
    register_module("convTranspose5", _convTranspose5);
    register_module("bn5", _bn5);
    register_module("dropout5", _dropout5);
    register_module("resNext6a", _resNext6a);
    register_module("resNext6b", _resNext6b);
    register_module("convTranspose6", _convTranspose6);
    register_module("bn6", _bn6);
    register_module("dropout6", _dropout6);
    register_module("resNext7a", _resNext7a);
    register_module("resNext7b", _resNext7b);
    register_module("convTranspose7", _convTranspose7);
    register_module("bn7", _bn7);
    register_module("conv8", _conv8);
    register_module("bn8", _bn8);
    register_module("conv8_Y", _conv8_Y);
    register_module("conv8_UV", _conv8_UV);

    auto* w = _conv8_Y->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);

    w = _conv8_UV->named_parameters(false).find("weight");
    if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
    torch::nn::init::normal_(*w, 0.0, 0.001);
}

std::tuple<torch::Tensor, torch::Tensor> FrameDecoder2Impl::forward(
    torch::Tensor x,
    torch::Tensor x0,
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor x3,
    torch::Tensor x4,
    torch::Tensor x5,
    torch::Tensor x6,
    double skipLevel
)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    int batchSize = x.sizes()[0];


    // Decoder
    x = torch::reshape(x, {batchSize, 128, 4, 4});
    x = _resNext1a(x);

//    printf("x.shape: [%ld %ld %ld %ld]\n", x.sizes()[0], x.sizes()[1], x.sizes()[2], x.sizes()[3]);
//    printf("x0.shape: [%ld %ld %ld %ld]\n", x0.sizes()[0], x0.sizes()[1], x0.sizes()[2], x0.sizes()[3]);
//    fflush(stdout);

    double s0 = std::clamp(std::abs(skipLevel), 0.0, 1.0);
    torch::Tensor loss = (1.0-s0)*(maxLoss(x, x0) + 10.0*l1_loss(x, x0));
    x = s0*x0 + (1.0-s0)*x;

    x = torch::tanh(_bn1(_convTranspose1(_resNext1b(x)))); // 5x5x512

    double s1 = std::clamp(skipLevel-1.0, 0.0, 1.0);
    loss += (s0-s1)*(maxLoss(x, x1) + 10.0*l1_loss(x, x1));
    x = s1*x1 + (1.0-s1)*x;

    x = torch::tanh(_bn2(_convTranspose2(_resNext2b(_resNext2a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 20x15x256

    double s2 = std::clamp(skipLevel-2.0, 0.0, 1.0);
    loss += (s1-s2)*(maxLoss(x, x2) + 10.0*l1_loss(x, x2));
    x = s2*x2 + (1.0-s2)*x;

    x = torch::tanh(_bn3(_convTranspose3(_resNext3b(_resNext3a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 40x30x256

    double s3 = std::clamp(skipLevel-3.0, 0.0, 1.0);
    loss += (s2-s3)*(maxLoss(x, x3) + 10.0*l1_loss(x, x3));
    x = s3*_dropout3(x3) + (1.0-s3)*x;

    x = torch::tanh(_bn4(_convTranspose4(_resNext4b(_resNext4a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 80x60x128

    double s4 = std::clamp(skipLevel-4.0, 0.0, 1.0);
    loss += (s3-s4)*(maxLoss(x, x4) + 10.0*l1_loss(x, x4));
    x = s4*_dropout4(x4) + (1.0-s4)*x;

    x = torch::tanh(_bn5(_convTranspose5(_resNext5b(_resNext5a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 160x120x64

    double s5 = std::clamp(skipLevel-5.0, 0.0, 1.0);
    loss += (s4-s5)*(maxLoss(x, x5) + 10.0*l1_loss(x, x5));
    x = s5*_dropout5(x5) + (1.0-s5)*x;

    x = torch::tanh(_bn6(_convTranspose6(_resNext6b(_resNext6a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 320x240x32

    double s6 = std::clamp(skipLevel-6.0, 0.0, 1.0);
    loss += (s5-s6)*(maxLoss(x, x6) + 10.0*l1_loss(x, x6));
    x = s6*_dropout6(x6) + (1.0-s6)*x;

    x = torch::tanh(_bn7(_convTranspose7(_resNext7b(_resNext7a(x)))));
    x = x.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)}); // 640x480x32

    x = torch::tanh(_bn8(_conv8(x)));

    torch::Tensor s_Y = 0.5f + 0.51f * torch::tanh(_conv8_Y(x));
        //x.index({Slice(), Slice(None, 8), Slice(), Slice()}))); // 640x480x1
    torch::Tensor s_UV = 0.51f * torch::tanh(_conv8_UV(x));
       // x.index({Slice(), Slice(8, None), Slice(), Slice()}))); // 640x480x2
    torch::Tensor s = torch::cat({s_Y, s_UV}, 1);

    return {s, loss};
}

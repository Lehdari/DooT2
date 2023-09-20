//
// Project: DooT2
// File: MultiLevelDecoderModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelDecoderModule.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


MultiLevelDecoderModuleImpl::MultiLevelDecoderModuleImpl(
    double level,
    int inputChannels,
    int outputChannels,
    int xUpscale,
    int yUpscale,
    int resBlockGroups,
    int resBlockScaling
) :
    _level          (level),
    _outputChannels (outputChannels),
    _xUpScale       (xUpscale),
    _yUpScale       (yUpscale),
    _resBlock1      (inputChannels, inputChannels*resBlockScaling, _outputChannels, resBlockGroups, 0.001),
    _resBlock2      (_outputChannels, _outputChannels*resBlockScaling, _outputChannels, resBlockGroups, 0.001),
    _convAux        (nn::Conv2dOptions(outputChannels, 8, {1, 1}).bias(false)),
    _bnAux          (nn::BatchNorm2dOptions(8)),
    _conv_Y         (nn::Conv2dOptions(8, 1, {1, 1})),
    _conv_UV        (nn::Conv2dOptions(8, 2, {1, 1}))
{
    register_module("resBlock1", _resBlock1);
    register_module("resBlock2", _resBlock2);
    register_module("convAux", _convAux);
    register_module("bnAux", _bnAux);
    register_module("conv_Y", _conv_Y);
    register_module("conv_UV", _conv_UV);

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

std::tuple<torch::Tensor, torch::Tensor> MultiLevelDecoderModuleImpl::forward(
    torch::Tensor x, double level, const torch::Tensor* imgPrev)
{
    using namespace torch::indexing;

    int outputWidth = x.sizes()[3]*_xUpScale;
    int outputHeight = x.sizes()[2]*_yUpScale;

    torch::Tensor y;
    if (level > _level) {
        auto originalType = x.scalar_type();
        x = tf::interpolate(x.to(kFloat32), tf::InterpolateFuncOptions()
            .size(std::vector<int64_t>{outputHeight, outputWidth})
            .mode(kBilinear)
            .align_corners(false))
            .to(originalType);

        x = _resBlock2(_resBlock1(x));

        // auxiliary image output
        y = gelu(_bnAux(_convAux(x)), "tanh");
        torch::Tensor y_Y = (imgPrev == nullptr ? 0.5f : 0.0f) + // don't add luminosity bias if image from previous layer is provided
            0.51f * torch::tanh(_conv_Y(y));
        torch::Tensor y_UV = 0.51f * torch::tanh(_conv_UV(y));
        y = torch::cat({y_Y, y_UV}, 1);

        // add to the previous image in case one is provided
        if (imgPrev != nullptr) {
            auto originalImgType = imgPrev->scalar_type();
            y = y + tf::interpolate(imgPrev->to(kFloat32), tf::InterpolateFuncOptions()
                .size(std::vector<int64_t>{outputHeight, outputWidth})
                .mode(kBilinear)
                .align_corners(false))
                .to(originalImgType);
        }
    }
    else {
        x = torch::zeros({x.sizes()[0], _outputChannels, outputHeight, outputWidth},
            TensorOptions().device(x.device()));
        y = torch::zeros({x.sizes()[0], 3, outputHeight, outputWidth},
            TensorOptions().device(x.device()));
    }

    return {x, y};
}

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
    int hiddenChannels,
    int outputChannels,
    int nGroups1,
    int nGroups2,
    int groupChannels1,
    int groupChannels2,
    int outputWidth,
    int outputHeight,
    int xUpscale,
    int yUpscale
) :
    _level              (level),
    _inputChannels      (inputChannels),
    _outputWidth        (outputWidth),
    _outputHeight       (outputHeight),
    _resNext1           (_inputChannels, hiddenChannels, nGroups1, groupChannels1),
    _resNext2           (hiddenChannels, outputChannels, nGroups2, groupChannels2),
    _convTranspose      (nn::ConvTranspose2dOptions(_inputChannels/2, _inputChannels/2, {xUpscale+2, yUpscale+2})
                         .stride({xUpscale, yUpscale}).bias(false)),
    _bnMain             (nn::BatchNorm2dOptions(_inputChannels/2)),
    _convSkip           (nn::Conv2dOptions(_inputChannels, outputChannels, {1,1}).bias(false)),
    _convAux            (nn::Conv2dOptions(outputChannels, 16, {3, 3}).bias(false).padding(1)),
    _bnAux              (nn::BatchNorm2dOptions(16)),
    _conv_Y             (nn::Conv2dOptions(16, 1, {1, 1})),
    _conv_UV            (nn::Conv2dOptions(16, 2, {1, 1}))
{
    register_module("resNext1", _resNext1);
    register_module("resNext2", _resNext2);
    register_module("convTranspose", _convTranspose);
    register_module("bnMain", _bnMain);
    register_module("convSkip", _convSkip);
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

std::tuple<torch::Tensor, torch::Tensor> MultiLevelDecoderModuleImpl::forward(torch::Tensor x, double level)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor y;
    if (level > _level) {
        y = x.index({Slice(), Slice(0, _inputChannels/2), Slice(), Slice()});
        y = torch::leaky_relu(_bnMain(_convTranspose(y)), leakyReluNegativeSlope);
        y = y.index({Slice(), Slice(), Slice(1, -1, None), Slice(1, -1, None)});

        auto originalType = x.scalar_type();
        x = x.index({Slice(), Slice(_inputChannels/2, None), Slice(), Slice()});
        x = tf::interpolate(x.to(kFloat32), tf::InterpolateFuncOptions()
            .size(std::vector<int64_t>{_outputHeight, _outputWidth})
            .mode(kBilinear)
            .align_corners(false)).to(originalType);

        x = torch::cat({x, y}, 1);
        x = _convSkip(x) + _resNext2(_resNext1(x));

        // auxiliary image output
        y = torch::leaky_relu(_bnAux(_convAux(x)), leakyReluNegativeSlope);
        torch::Tensor y_Y = 0.5f + 0.51f * torch::tanh(_conv_Y(y));
        torch::Tensor y_UV = 0.51f * torch::tanh(_conv_UV(y));
        y = torch::cat({y_Y, y_UV}, 1);
    }
    else
        y = torch::zeros({x.sizes()[0], 3, _outputHeight, _outputWidth}, TensorOptions().device(x.device()));

    return {x, y};
}

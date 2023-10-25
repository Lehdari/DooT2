//
// Project: DooT2
// File: FourierConv2d.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/FourierConv2d.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


FourierConv2dImpl::FourierConv2dImpl(
    int inputChannels,
    int outputChannels,
    double globalChannelRatio,
    int groups,
    double normalInitializationStd
):
    _localInputChannels     (inputChannels*(1.0-globalChannelRatio)),
    _globalInputChannels    (inputChannels-_localInputChannels),
    _localOutputChannels    (outputChannels*(1.0-globalChannelRatio)),
    _globalOutputChannels   (outputChannels-_localOutputChannels),
    _convLocal              (nn::Conv2dOptions(_localInputChannels, _localOutputChannels, {3,3}).bias(false)
                             .groups(groups).padding({1,1})),
    _convLocalGlobal        (nn::Conv2dOptions(_localInputChannels, _globalOutputChannels, {3,3}).bias(false)
                             .groups(groups).padding({1,1})),
    _convGlobalLocal        (nn::Conv2dOptions(_globalInputChannels, _localOutputChannels, {3,3}).bias(false)
                             .groups(groups).padding({1,1})),
    _convGlobal1            (nn::Conv2dOptions(_globalInputChannels, _globalInputChannels/2, {1,1}).bias(false)),
    _convGlobal2            (nn::Conv2dOptions(_globalInputChannels, _globalInputChannels, {1,1}).bias(false)),
    _convGlobal3            (nn::Conv2dOptions(_globalInputChannels/2, _globalOutputChannels, {1,1}).bias(false)),
    _bnGlobal1              (nn::BatchNorm2dOptions(_globalInputChannels/2)),
    _bnGlobal2              (nn::BatchNorm2dOptions(_globalInputChannels)),
    _bnLocalMerge           (nn::BatchNorm2dOptions(_localOutputChannels)),
    _bnGlobalMerge          (nn::BatchNorm2dOptions(_globalOutputChannels))
{
    if (normalInitializationStd > 0.0) {
        auto* w = _convLocal->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
        w = _convLocalGlobal->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
        w = _convGlobalLocal->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
        w = _convGlobal3->named_parameters(false).find("weight");
        if (w == nullptr) throw std::runtime_error("Unable to find layer weights");
        torch::nn::init::normal_(*w, 0.0, normalInitializationStd);
    }

    register_module("convLocal", _convLocal);
    register_module("convLocalGlobal", _convLocalGlobal);
    register_module("convGlobalLocal", _convGlobalLocal);
    register_module("convGlobal1", _convGlobal1);
    register_module("convGlobal2", _convGlobal2);
    register_module("convGlobal3", _convGlobal3);
    register_module("bnGlobal1", _bnGlobal1);
    register_module("bnGlobal2", _bnGlobal2);
    register_module("bnLocalMerge", _bnLocalMerge);
    register_module("bnGlobalMerge", _bnGlobalMerge);
}

torch::Tensor FourierConv2dImpl::forward(const torch::Tensor& x)
{
    using namespace torch::indexing;

    auto dtype = x.dtype();

    torch::Tensor local = x.index({Slice(), Slice(None, _localInputChannels), Slice(), Slice()});
    torch::Tensor global = x.index({Slice(), Slice(_localInputChannels, None), Slice(), Slice()});
    assert(global.sizes()[1] == _globalInputChannels);

    torch::Tensor local2 = gelu(_bnLocalMerge(_convLocal(local) + _convGlobalLocal(global)), "tanh");

    // Spectral transform path
    torch::Tensor global2 = gelu(_bnGlobal1(_convGlobal1(global)), "tanh");
    // 2D FFT
    global = torch::fft_rfft2(global2.to(torch::kFloat32)); // [B * C/2 * H * W/2] complex tensor, C: _globalInputChannels
    // concatenate the real and imaginary parts along channel dimension into [B * C * H * W/2] real tensor
    global = torch::cat({real(global), imag(global)}, 1);
    global = gelu(_bnGlobal2(_convGlobal2(global)));
    // transform back to a complex tensor of shape [B * C/2 * H * W/2]
    global = torch::complex(
        global.index({Slice(), Slice(None, _globalInputChannels/2), Slice(), Slice()}).to(torch::kFloat32),
        global.index({Slice(), Slice(_globalInputChannels/2, None), Slice(), Slice()}).to(torch::kFloat32));
    // 2D IFFT
    global = torch::fft_irfft2(global, { global2.sizes()[2], global2.sizes()[3] });
    global = _convGlobal3(global + global2);

    global2 = gelu(_bnGlobalMerge(global + _convLocalGlobal(local)), "tanh");

    return torch::cat({local2, global2}, 1);
}

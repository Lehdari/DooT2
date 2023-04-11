//
// Project: DooT2
// File: ResNeXtModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/ResNeXtModule.hpp"


using namespace ml;
using namespace torch;


ResNeXtModuleImpl::ResNeXtModuleImpl(int nInputChannels, int nOutputChannels, int nGroups, int nGroupChannels) :
    _nInputChannels     (nInputChannels),
    _nOutputChannels    (nOutputChannels),
    _nGroups            (nGroups),
    _nGroupChannels     (nGroupChannels < 0 ? _nOutputChannels : nGroupChannels*_nGroups),
    _conv1              (nn::Conv2dOptions(_nInputChannels, _nGroupChannels, {1,1}).bias(false)),
    _bn1                (nn::BatchNorm2dOptions(_nGroupChannels)),
    _conv2              (nn::Conv2dOptions(_nGroupChannels, _nGroupChannels, {3,3})
                         .bias(false).groups(_nGroups).padding(1)),
    _bn2                (nn::BatchNorm2dOptions(_nGroupChannels)),
    _conv3              (nn::Conv2dOptions(_nGroupChannels, _nOutputChannels, {1,1}).bias(false)),
    _bn3                (nn::BatchNorm2dOptions(_nOutputChannels))
{
    register_module("conv1", _conv1);
    register_module("bn1", _bn1);
    register_module("conv2", _conv2);
    register_module("bn2", _bn2);
    register_module("conv3", _conv3);
    register_module("bn3", _bn3);
}

torch::Tensor ResNeXtModuleImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    constexpr double leakyReluNegativeSlope = 0.01;

    torch::Tensor skip = x;
    if (_nOutputChannels < _nInputChannels) {
        skip = x.index({Slice(), Slice(None, _nOutputChannels), Slice(), Slice()});
    }
    else if (_nOutputChannels > _nInputChannels) {
        int nRepeats = _nOutputChannels / _nInputChannels;
        if (_nOutputChannels % _nInputChannels > 0)
            ++nRepeats;
        skip = skip.repeat({1, nRepeats, 1, 1});
        skip = skip.index({Slice(), Slice(None, _nOutputChannels), Slice(), Slice()});
    }

    x = torch::tanh(_bn1(_conv1(x)));
    x = torch::tanh(_bn2(_conv2(x)));
    x = torch::tanh(_bn3(_conv3(x) + skip));

    return x;
}

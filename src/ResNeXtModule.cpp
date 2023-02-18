//
// Project: DooT2
// File: ResNeXtModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ResNeXtModule.hpp"


using namespace torch;


ResNeXtModuleImpl::ResNeXtModuleImpl(int nInputChannels, int nGroupChannels, int nGroups, int nOutputChannels) :
    _nInputChannels     (nInputChannels),
    _nOutputChannels    (nOutputChannels),
    _nGroups            (nGroups),
    _convFinal          (nn::Conv2dOptions(nGroupChannels*_nGroups, nOutputChannels, {1,1}).bias(false)),
    _bnFinal            (nn::BatchNorm2dOptions(nOutputChannels))
{
    // Add 1x1, 3x3 conv layers and batch norms for each group
    _groups.reserve(_nGroups);
    for (int i=0; i<_nGroups; ++i) {
        _groups.emplace_back(
            nn::Conv2d(nn::Conv2dOptions(nInputChannels, nGroupChannels, {1,1}).bias(false)),
            nn::BatchNorm2d(nn::BatchNorm2dOptions(nGroupChannels)),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.01)),
            nn::Conv2d(nn::Conv2dOptions(nGroupChannels, nGroupChannels, {3,3}).bias(false).padding(1)),
            nn::BatchNorm2d(nn::BatchNorm2dOptions(nGroupChannels)),
            nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.01))
        );

        register_module("group" + std::to_string(i), _groups.back());
    }

    register_module("convFinal", _convFinal);
    register_module("bnFinal", _bnFinal);

    // Allocate group outputs
    _groupOutputs.resize(_nGroups);
}

torch::Tensor ResNeXtModuleImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    for (int i=0; i<_nGroups; ++i) {
        _groupOutputs[i] = _groups[i]->forward(x);
    }

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

    return skip + torch::leaky_relu(_bnFinal(_convFinal(cat(_groupOutputs, 1))), 0.01);
}

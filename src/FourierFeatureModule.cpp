//
// Project: DooT2
// File: FourierFeatureModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FourierFeatureModule.hpp"


using namespace torch;


FourierFeatureModuleImpl::FourierFeatureModuleImpl(int nInputChannels, int nFeatures, double minStd, double maxStd) :
    _nFeatures  (nFeatures),
    _conv       (nn::Conv2dOptions(nInputChannels, 6*_nFeatures, {1, 1}))
{
    using namespace torch::indexing;

    register_module("conv", _conv);

    auto* w = _conv->named_parameters(false).find("weight");
    if (w == nullptr)
        throw std::runtime_error("Unable to find layer weights");

    // Initialization
    torch::nn::init::normal_(*w);
    // Initialize the basis matrix square section with exponentially increasing frequencies
    double scaleBase = std::pow(minStd/maxStd, 1.0/(nFeatures-1));
    for (int i=0; i<_nFeatures; ++i) {
        double std = maxStd * std::pow(scaleBase, (double)i);
        torch::nn::init::normal_(w->index({Slice(6*i, 6*i+2, None), "..."}), 0.0, std);
        torch::nn::init::normal_(w->index({Slice(6*i+3, 6*i+5, None), "..."}), 0.0, std);
    }
}

torch::Tensor FourierFeatureModuleImpl::forward(torch::Tensor x)
{
    using namespace torch::indexing;

    auto device = x.device();
    int batchSize = x.sizes()[0];
    int height = x.sizes()[2];
    int width = x.sizes()[3];

    auto mg = torch::meshgrid({
        torch::linspace(-0.75f, 0.75f, height).to(device),
        torch::linspace(-1.0f, 1.0f, width).to(device)}, "ij");
    auto a = torch::cat({
        mg[0].unsqueeze(2).unsqueeze(3),
        mg[1].unsqueeze(2).unsqueeze(3),
        torch::ones({height, width, 1, 1}, TensorOptions().device(device))
    }, 2);

    auto y = _conv(x).permute({0, 2, 3, 1}).reshape({batchSize, height, width, -1, 3});
    y = torch::matmul(y, a).reshape({batchSize, height, width, -1, 2});
    y.index_put_({Slice(), Slice(), Slice(), Slice(), 0},
        torch::cos(y.index({Slice(), Slice(), Slice(), Slice(), 0}).clone()));
    y.index_put_({Slice(), Slice(), Slice(), Slice(), 1},
        torch::sin(y.index({Slice(), Slice(), Slice(), Slice(), 1}).clone()));
    y = y.reshape({batchSize, height, width, -1, 1}).squeeze().permute({0, 3, 1, 2});

    x = torch::cat({x, y}, 1);

    return x;
}

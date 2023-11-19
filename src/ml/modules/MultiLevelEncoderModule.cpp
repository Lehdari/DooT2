//
// Project: DooT2
// File: MultiLevelEncoderModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelEncoderModule.hpp"
#include "ml/modules/ResNetConvBlock.hpp"
#include "ml/modules/ResNetFourierConvBlock.hpp"
#include "ml/modules/ViTBlock.hpp"


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;


MultiLevelEncoderModuleImpl::MultiLevelEncoderModuleImpl(
    const std::string& resBlockConfig,
    double level,
    int inputChannels,
    int outputChannels,
    int xDownScale,
    int yDownScale,
    int resBlockGroups,
    int resBlockScaling,
    int transformerHeads,
    int transformerHeadDim
) :
    _resBlockConfig         (resBlockConfig),
    _level                  (level),
    _outputChannels         (outputChannels),
    _downscaleResBlock      (inputChannels, _outputChannels*resBlockScaling, _outputChannels, xDownScale, yDownScale,
                             resBlockGroups, true, 0.0, 0.001),
    _conv1Aux               (nn::Conv2dOptions(3, _outputChannels, {1, 1}).bias(false)),
    _bn1Aux                 (nn::BatchNorm2dOptions(_outputChannels))
{
    // setup residual blocks
    for (const auto& blockType : _resBlockConfig) {
        switch (blockType) {
            case 'T': {
                _resBlocks->push_back(ViTBlock(
                    _outputChannels, transformerHeads, transformerHeadDim, _outputChannels*resBlockScaling));
            }   break;
            case 'C': {
                _resBlocks->push_back(ResNetConvBlock(
                    _outputChannels, _outputChannels*resBlockScaling,
                    _outputChannels, resBlockGroups));
            }   break;
            case 'F': {
                _resBlocks->push_back(ResNetFourierConvBlock(
                    _outputChannels, _outputChannels*resBlockScaling,
                    _outputChannels, resBlockGroups));
            }   break;
            default: {
                fprintf(stderr, "Unknown residual block type: %c, omitting...\n", blockType);
            }   break;
        }
    }

    register_module("downscaleResBlock", _downscaleResBlock);
    register_module("conv1Aux", _conv1Aux);
    register_module("bn1Aux", _bn1Aux);
    register_module("resBlocks", _resBlocks);
}

torch::Tensor MultiLevelEncoderModuleImpl::forward(const Tensor& main, const Tensor& aux, double level)
{
    torch::Tensor x, y;
    if (level > _level) {
        x = _downscaleResBlock(main);
        y = gelu(_bn1Aux(_conv1Aux(aux)));
        float w = (float)std::clamp(_level+1.0-level, 0.0, 1.0);
        x = w*y + (1.0f-w)*x;

        // call res blocks
        int resBlockId = 0;
        for (const auto& resBlock : *_resBlocks) {
            switch (_resBlockConfig[resBlockId]) {
                case 'T': { x = resBlock->as<ViTBlock>()->forward(x); }   break;
                case 'C': { x = resBlock->as<ResNetConvBlock>()->forward(x); }   break;
                case 'F': { x = resBlock->as<ResNetFourierConvBlock>()->forward(x); }   break;
                default: break;
            }
            ++resBlockId;
        }
    }
    else if (level > _level-1.0) {
        x = gelu(_bn1Aux(_conv1Aux(aux)));

        // call res blocks
        int resBlockId = 0;
        for (const auto& resBlock : *_resBlocks) {
            switch (_resBlockConfig[resBlockId]) {
                case 'T': { x = resBlock->as<ViTBlock>()->forward(x); }   break;
                case 'C': { x = resBlock->as<ResNetConvBlock>()->forward(x); }   break;
                case 'F': { x = resBlock->as<ResNetFourierConvBlock>()->forward(x); }   break;
                default: break;
            }
            ++resBlockId;
        }
    }
    else
        x = torch::zeros({aux.sizes()[0], _outputChannels, aux.sizes()[2], aux.sizes()[3]},
            torch::TensorOptions().device(main.device()).dtype(main.dtype()));

    return x;
}

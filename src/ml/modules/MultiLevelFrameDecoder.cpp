//
// Project: DooT2
// File: MultiLevelFrameDecoder.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/modules/MultiLevelFrameDecoder.hpp"


using namespace ml;
using namespace torch;
using namespace torch::indexing;
namespace tf = torch::nn::functional;


MultiLevelFrameDecoderImpl::MultiLevelFrameDecoderImpl() :
    _linear1            (nn::LinearOptions(2048, 2048).bias(false)),
    _bn1                (nn::BatchNorm1dOptions(2048)),
    _pRelu1             (nn::PReLUOptions().num_parameters(2048).init(0.01)),
    _decoder1           (-1.0, 128, 256, 512, 1024, 16, 16, 32, 64, 5, 5, ExpandingArray<2>{2, 2}),
    _decoder2           (0.0, 1024, 512, 512, 512, 16, 16, 64, 64, 10, 15,
                         ExpandingArray<2>{5, 4}, ExpandingArray<2>{3, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder3           (1.0, 512, 512, 512, 512, 16, 16, 64, 64, 20, 15,
                         ExpandingArray<2>{3, 4}, ExpandingArray<2>{1, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder4           (2.0, 512, 512, 512, 256, 16, 16, 64, 64, 40, 30,
                         ExpandingArray<2>{4, 4}, ExpandingArray<2>{2, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder5           (3.0, 256, 256, 256, 128, 8, 8, 64, 64, 80, 60,
                         ExpandingArray<2>{4, 4}, ExpandingArray<2>{2, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder6           (4.0, 128, 128, 128, 64, 8, 8, 32, 32, 160, 120,
                         ExpandingArray<2>{4, 4}, ExpandingArray<2>{2, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder7           (5.0, 64, 64, 64, 32, 8, 8, 16, 16, 320, 240,
                         ExpandingArray<2>{4, 4}, ExpandingArray<2>{2, 2}, Slice(1, -1, None), Slice(1, -1, None)),
    _decoder8           (6.0, 32, 32, 32, 32, 8, 8, 8, 8, 640, 480,
                         ExpandingArray<2>{4, 4}, ExpandingArray<2>{2, 2}, Slice(1, -1, None), Slice(1, -1, None))
{
    register_module("linear0_1", _linear1);
    register_module("bn1", _bn1);
    register_module("pRelu1", _pRelu1);
    register_module("decoder1", _decoder1);
    register_module("decoder2", _decoder2);
    register_module("decoder3", _decoder3);
    register_module("decoder4", _decoder4);
    register_module("decoder5", _decoder5);
    register_module("decoder6", _decoder6);
    register_module("decoder7", _decoder7);
    register_module("decoder8", _decoder8);
}

MultiLevelImage MultiLevelFrameDecoderImpl::forward(torch::Tensor x, double level)
{
    using namespace torch::indexing;

    int batchSize = x.sizes()[0];


    // Decoder
    // Linear layer
    x = _pRelu1(_bn1(_linear1(x)));
    x = torch::reshape(x, {batchSize, 128, 4, 4});

    MultiLevelImage img;
    img.level = level;
    std::tie(x, img.img0) = _decoder1(x, level);
    std::tie(x, img.img1) = _decoder2(x, level);
    std::tie(x, img.img2) = _decoder3(x, level);
    std::tie(x, img.img3) = _decoder4(x, level);
    std::tie(x, img.img4) = _decoder5(x, level);
    std::tie(x, img.img5) = _decoder6(x, level);
    std::tie(x, img.img6) = _decoder7(x, level);
    std::tie(x, img.img7) = _decoder8(x, level);

    return img;
}

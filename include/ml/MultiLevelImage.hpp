//
// Project: DooT2
// File: MultiLevelImage.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>


namespace ml {

struct MultiLevelImage {
    torch::Tensor   img0; // 5x5
    torch::Tensor   img1; // 10x15
    torch::Tensor   img2; // 20x15
    torch::Tensor   img3; // 40x30
    torch::Tensor   img4; // 80x60
    torch::Tensor   img5; // 160x120
    torch::Tensor   img6; // 320x240
    torch::Tensor   img7; // 640x480
    double          level   {0};
};

} // namespace ml

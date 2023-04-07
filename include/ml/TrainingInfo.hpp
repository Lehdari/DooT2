//
// Project: DooT2
// File: TrainingInfo.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "util/Image.hpp"
#include "util/SingleBuffer.hpp"
#include "util/TimeSeries.hpp"

#include <unordered_map>


namespace ml {

struct TrainingInfo;


// Struct for storing information about training,
// used for communication between the training process and GUI
struct TrainingInfo {
    using ImageMap = std::unordered_map<std::string, SingleBuffer<Image<float>>>;

    SingleBuffer<TimeSeries>    timeSeries;
    ImageMap                    images;
};

} // namespace ml

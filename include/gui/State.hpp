//
// Project: DooT2
// File: State.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "gui/ImageRelay.hpp"

#include <map>
#include <string>


class TimeSeries;
template <typename T_Data>
class SingleBuffer;


namespace gui {

struct State {
    using TimeSeriesMap = std::unordered_map<std::string, SingleBuffer<TimeSeries>*>;
    using ImageRelayMap = std::map<std::string, gui::ImageRelay>;
    using CallbackMap = std::unordered_map<std::string, std::function<void(const State&)>>;

    // Sources for time series data
    TimeSeriesMap   timeSeries;

    CallbackMap     callbacks;

    // Game window state
    gut::Texture    frameTexture;

    // Training Images frame
    ImageRelayMap   modelImageRelays;

    // Training status
    enum class TrainingStatus : int32_t {
        STOPPED = 0,
        ONGOING = 1,
        PAUSED  = 2
    }               trainingStatus          {TrainingStatus::STOPPED};

    char            experimentName[256]     {"ex_{time}_{version}"};
    std::string     experimentBase          {""};
    nlohmann::json  baseExperimentConfig;
    std::string     modelTypeName           {"AutoEncoderModel"}; // type name of the model to be trained
};

} // namespace gui

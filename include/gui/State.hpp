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
    }               trainingStatus              {TrainingStatus::STOPPED};

    // Experiment configuration parameters
    std::string     experimentName              {"ex_{time}_{version}"};
    std::string     experimentBase;
    nlohmann::json  baseExperimentConfig;
    enum class TrainingTask : int32_t {
        FRAME_ENCODING = 0,
        AGENT_POLICY = 1
    }               trainingTask                {TrainingTask::FRAME_ENCODING};
    std::string     modelTypeName               {"AutoEncoderModel"}; // type name of the model to be trained
    int32_t         evaluationInterval          {16}; // the performance of the model will be evaluated between this many epochs
    bool            useSequenceCache            {false}; // Flag on whether to use cached frames on the frame encoding task
    std::string     sequenceCachePath           {"sequence_cache/"};
    int32_t         nCachedSequences            {1}; // number of different cached sequences to use when using sequence cache
    int32_t         sequenceCacheRecordInterval {64}; // replace the oldest training sequence batch between this many epochs
    bool            gridSearch                  {false};
    nlohmann::json  gridSearchModelConfigParams;
};

} // namespace gui

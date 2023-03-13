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

    // Sources for time series data
    TimeSeriesMap   _timeSeries;

    // Game window state
    gut::Texture    _frameTexture;

    // Training Images frame
    ImageRelayMap   _modelImageRelays;
};

} // namespace gui

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


namespace gui {

struct State {
    using TimeSeriesMap = std::map<std::string, std::pair<const std::vector<double>*, bool>>; // bool: is the series active (displayed)
    using ImageRelayMap = std::map<std::string, gui::ImageRelay>;

    // Plot window state
    TimeSeriesMap   _plotTimeSeriesVectors;
    bool            _lossPlotAutoFit        {false};
    bool            _lossPlotTimeMode       {false};
    char            _plotFileName[256]      {"loss.plot"};

    // Frame window state
    bool            _showFrame              {true};
    gut::Texture    _frameTexture;

    // Training Images frame
    bool            _showTrainingImages     {true};
    ImageRelayMap   _modelImageRelays;
    std::string     _currentModelImage;
};

} // namespace gui

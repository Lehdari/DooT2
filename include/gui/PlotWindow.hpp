//
// Project: DooT2
// File: PlotWindow.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Window.hpp"

#include <string>
#include <map>
#include <vector>


namespace gui {

class PlotWindow : public Window {
public:
    PlotWindow(std::set<int>* activeIds, State* guiState, int id = -1) :
        Window(this, guiState, activeIds, id)
    {}

    void update() override;
    void render(ml::Trainer* trainer) override;
    void applyConfig(const nlohmann::json& config) override;
    nlohmann::json getConfig() const override;

private:
    using ActiveSeriesMap = std::map<std::string, std::map<std::string, bool>>;

    ActiveSeriesMap _activeSeries;
    std::string     _activeSource       {"training"};
    bool            _lossPlotAutoFit    {false};
    bool            _lossPlotTimeMode   {false};
    char            _plotFileName[256]  {"loss.plot"};
};

};

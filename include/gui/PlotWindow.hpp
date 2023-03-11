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
    PlotWindow(std::set<int>* activeIds) :
        Window(activeIds)
    {}

    virtual void update(gui::State* guiState) override;
    virtual void render(Trainer* trainer, Model* model, gui::State* guiState) override;

private:
    using ActiveSeriesMap = std::map<std::string, std::map<std::string, bool>>;

    ActiveSeriesMap _activeSeries;
    std::string     _activeSource       {"training"};
    bool            _lossPlotAutoFit    {false};
    bool            _lossPlotTimeMode   {false};
    char            _plotFileName[256]  {"loss.plot"};
};

};

//
// Project: DooT2
// File: PlotWindow.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/PlotWindow.hpp"
#include "gui/State.hpp"
#include "Model.hpp"

#include "implot.h"


void gui::PlotWindow::update(gui::State* guiState)
{
    // Update the available time series
    ActiveSeriesMap newActiveSeries;
    for (auto& [sourceName, timeSeries] : guiState->_timeSeries) {
        auto timeSeriesHandle = timeSeries->read();
        auto seriesNames = timeSeriesHandle->getSeriesNames();
        for (auto& seriesName : seriesNames) {
            if (seriesName == "time") // time is dedicated to be used in x-axis
                continue;
            bool active = false;
            // Use existing value in case one exists
            if (_activeSeries.contains(sourceName) && _activeSeries[sourceName].contains(seriesName))
                active = _activeSeries[sourceName][seriesName];
            newActiveSeries[sourceName][seriesName] = active;
        }
    }
    _activeSeries = std::move(newActiveSeries);

    // In case the active source has been removed, pick the first one available
    if (!_activeSeries.contains(_activeSource)) {
        if (_activeSeries.empty())
            _activeSource = "";
        else
            _activeSource = _activeSeries.begin()->first;
    }
}

void gui::PlotWindow::render(Trainer* trainer, Model* model, gui::State* guiState)
{
    if (!_open) return;
    if (ImGui::Begin(("Plotting " + std::to_string(_id)).c_str(), &_open)) {
        ImVec2 plotWindowSize = ImGui::GetWindowSize();

        // Time series source selector
        ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
        if (ImGui::BeginCombo("##SourceSelector", _activeSource.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto& [sourceName, timeSeriesSource] : _activeSeries) {
                bool isSelected = (_activeSource == sourceName);
                if (ImGui::Selectable(sourceName.c_str(), isSelected))
                    _activeSource = sourceName;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Time series selector
        ImGui::SameLine();
        ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
        if (ImGui::BeginCombo("##PlotSelector", "Select Plots")) {
            for (auto& [name, active] : _activeSeries[_activeSource]) {
                ImGui::Checkbox(name.c_str(), &active);
            }
            ImGui::EndCombo();
        }

#if 0 // TODO move elsewhere
        ImGui::SameLine();
        ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
        ImGui::InputText("##PlotFileName", _plotFileName, 255);

        // Save button
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            auto timeSeriesReadHandle = model->timeSeries.read();
            auto plotJson = timeSeriesReadHandle->toJson();
            std::ofstream plotFile(_plotFileName);
            plotFile << plotJson;
            plotFile.close();
            printf("Plot saved!\n");
        }
#endif

        ImGui::SameLine();
        ImGui::Checkbox("Autofit plot", &_lossPlotAutoFit);
        ImGui::SameLine();
        ImGui::Checkbox("Time on X-axis", &_lossPlotTimeMode);

        // Loss plot
        if (ImPlot::BeginPlot("##Plot", ImVec2(-1, -1))) {
            auto lossPlotAxisFlags = _lossPlotAutoFit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
            ImPlot::SetupAxes(_lossPlotTimeMode ? "Training Time (s)" : "Training Step", "",
                lossPlotAxisFlags, lossPlotAxisFlags);

            for (auto& [sourceName, timeSeriesSource] : _activeSeries) {
                auto timeSeriesHandle = guiState->_timeSeries[sourceName]->read();
                auto& timeVector = *timeSeriesHandle->getSeriesVector<double>("time");
                for (auto& [seriesName, active] : timeSeriesSource) {
                    if (active) {
                        auto& seriesVector = *timeSeriesHandle->getSeriesVector<double>(seriesName);
                        if (_lossPlotTimeMode) {
                            ImPlot::PlotLine(seriesName.c_str(), timeVector.data(), seriesVector.data(),
                                (int)seriesVector.size());
                        } else {
                            ImPlot::PlotLine(seriesName.c_str(), seriesVector.data(), (int)seriesVector.size());
                        }
                    }
                }
            }
            ImPlot::EndPlot();
        }

        ImGui::End();
    }
    else
        ImGui::End();
}

void gui::PlotWindow::applyConfig(const nlohmann::json& config)
{
    for (auto& [sourceName, timeSeriesSource] : config["activeSeries"].items()) {
        for (auto& [name, active] : timeSeriesSource.items()) {
            _activeSeries[sourceName][name] = active;
        }
    }
    _activeSource = config["activeSource"].get<std::string>();
    if (!_activeSeries.contains(_activeSource)) {
        if (_activeSeries.empty())
            _activeSource = "";
        else
            _activeSource = _activeSeries.begin()->first;
    }
    _lossPlotAutoFit = config["lossPlotAutoFit"].get<bool>();
    _lossPlotTimeMode = config["lossPlotTimeMode"].get<bool>();
}

nlohmann::json gui::PlotWindow::getConfig() const
{
    nlohmann::json config;
    nlohmann::json activeSeriesConfig;
    for (auto& [sourceName, timeSeriesSource] : _activeSeries) {
        nlohmann::json timeSeriesSourceConfig;
        for (auto& [seriesName, active] : timeSeriesSource) {
            timeSeriesSourceConfig[seriesName] = active;
        }
        activeSeriesConfig[sourceName] = timeSeriesSourceConfig;
    }
    config["activeSeries"] = activeSeriesConfig;
    config["activeSource"] = _activeSource;
    config["lossPlotAutoFit"] = _lossPlotAutoFit;
    config["lossPlotTimeMode"] = _lossPlotTimeMode;
    return config;
}

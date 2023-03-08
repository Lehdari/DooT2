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

#include <fstream>


void gui::PlotWindow::render(Trainer* trainer, Model* model, gui::State* guiState) const
{
    ImGui::Begin("Plot");
    ImVec2 plotWindowSize = ImGui::GetWindowSize();
    ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
    if (ImGui::BeginCombo("##PlotSelector", "Select Plots")) {
        for (auto& [name, timeSeries] : guiState->_plotTimeSeriesVectors) {
            ImGui::Checkbox(name.c_str(), &timeSeries.second);
        }
        ImGui::EndCombo();
    }


    ImGui::SameLine();
    ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
    ImGui::InputText("##PlotFileName", guiState->_plotFileName, 255);

    // Save button
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        auto timeSeriesReadHandle = model->timeSeries.read();
        auto plotJson = timeSeriesReadHandle->toJson();
        std::ofstream plotFile(guiState->_plotFileName);
        plotFile << plotJson;
        plotFile.close();
        printf("Plot saved!\n");
    }

    // TODO load timeseries button here

    ImGui::SameLine();
    ImGui::Checkbox("Autofit plot", &guiState->_lossPlotAutoFit);
    ImGui::SameLine();
    ImGui::Checkbox("Time on X-axis", &guiState->_lossPlotTimeMode);

    {   // Loss plot
        auto timeSeriesReadHandle = model->timeSeries.read();

        if (ImPlot::BeginPlot("##Plot", ImVec2(-1, -1))) {
            auto lossPlotAxisFlags = guiState->_lossPlotAutoFit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
            ImPlot::SetupAxes(guiState->_lossPlotTimeMode ? "Training Time (s)" : "Training Step", "",
                lossPlotAxisFlags, lossPlotAxisFlags);

            auto& timeVector = *timeSeriesReadHandle->getSeriesVector<double>("time");
            for (auto& [name, timeSeries] : guiState->_plotTimeSeriesVectors) {
                if (timeSeries.second) {
                    if (guiState->_lossPlotTimeMode) {
                        ImPlot::PlotLine(name.c_str(), timeVector.data(), timeSeries.first->data(),
                            (int) timeSeries.first->size());
                    }
                    else {
                        ImPlot::PlotLine(name.c_str(), timeSeries.first->data(), (int)timeSeries.first->size());
                    }
                }
            }

            ImPlot::EndPlot();
        }
    }

    ImGui::End(); // Plot
}

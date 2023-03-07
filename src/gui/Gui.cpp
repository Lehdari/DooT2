//
// Project: DooT2
// File: Gui.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/Gui.hpp"
#include "Trainer.hpp"
#include "Model.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "implot.h"
#include <SDL.h>


using namespace gui;


Gui::~Gui()
{
    // Destroy imgui
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
}

void Gui::init(SDL_Window* window, SDL_GLContext* glContext)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    _guiState._frameTexture.create(doomGame.getScreenWidth(), doomGame.getScreenHeight(),
        GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE);

    // Initialize imgui
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForOpenGL(window, glContext);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImPlot::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void Gui::update(Model* model)
{
    // Update the plot time series map
    _guiState._plotTimeSeriesVectors.clear();
    {
        auto timeSeriesHandle = model->timeSeries.read();
        auto timeSeriesNames = timeSeriesHandle->getSeriesNames();
        for (const auto& name : timeSeriesNames) {
            if (name == "time") // "time" is dedicated to be used as plot x-coordinates when time mode is selected
                continue;
            auto* seriesVector = timeSeriesHandle->getSeriesVector<double>(name);
            if (seriesVector == nullptr) // the series was not of double type, skip it
                continue;
            _guiState._plotTimeSeriesVectors.emplace(name, std::make_pair(seriesVector, false));
        }
    }

    // Update the model images map
    _guiState._modelImageRelays.clear();
    for (auto& [name, imageBuffer] : model->images) {
        _guiState._modelImageRelays.emplace(name, &imageBuffer);
    }
}

void Gui::render(SDL_Window* window, Trainer* trainer, Model* model)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    imGuiNewFrame(window);

    // Make the entire window dockable
    ImGui::DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

    ImGui::Begin("Plot");
    ImVec2 plotWindowSize = ImGui::GetWindowSize();
    ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
    if (ImGui::BeginCombo("##PlotSelector", "Select Plots")) {
        for (auto& [name, timeSeries] : _guiState._plotTimeSeriesVectors) {
            ImGui::Checkbox(name.c_str(), &timeSeries.second);
        }
        ImGui::EndCombo();
    }


    ImGui::SameLine();
    ImGui::SetNextItemWidth(plotWindowSize.x * 0.5f - ImGui::GetFontSize() * 12.5f);
    ImGui::InputText("##PlotFileName", _guiState._plotFileName, 255);

    // Save button
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        auto timeSeriesReadHandle = model->timeSeries.read();
        auto plotJson = timeSeriesReadHandle->toJson();
        std::ofstream plotFile(_guiState._plotFileName);
        plotFile << plotJson;
        plotFile.close();
        printf("Plot saved!\n");
    }

    // TODO load timeseries button here

    ImGui::SameLine();
    ImGui::Checkbox("Autofit plot", &_guiState._lossPlotAutoFit);
    ImGui::SameLine();
    ImGui::Checkbox("Time on X-axis", &_guiState._lossPlotTimeMode);

    {   // Loss plot
        auto timeSeriesReadHandle = model->timeSeries.read();

        if (ImPlot::BeginPlot("##Plot", ImVec2(-1, -1))) {
            auto lossPlotAxisFlags = _guiState._lossPlotAutoFit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
            ImPlot::SetupAxes(_guiState._lossPlotTimeMode ? "Training Time (s)" : "Training Step", "",
                lossPlotAxisFlags, lossPlotAxisFlags);

            auto& timeVector = *timeSeriesReadHandle->getSeriesVector<double>("time");
            for (auto& [name, timeSeries] : _guiState._plotTimeSeriesVectors) {
                if (timeSeries.second) {
                    if (_guiState._lossPlotTimeMode) {
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

    if (_guiState._showFrame && ImGui::Begin("Frame", &_guiState._showFrame)) {
        {
            auto frameHandle = trainer->getFrameReadHandle();
            _guiState._frameTexture.updateFromBuffer(frameHandle->data(), GL_BGRA);
        }
        ImGui::Image((void*)(intptr_t)_guiState._frameTexture.id(),
            ImVec2(doomGame.getScreenWidth(), doomGame.getScreenHeight()),
            ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
        ImGui::End(); // Frame
    }
    else {
        ImGui::End(); // Frame
    }

    if (_guiState._showTrainingImages && ImGui::Begin("Training Images", &_guiState._showTrainingImages)) {
        if (ImGui::BeginCombo("##combo", _guiState._currentModelImage.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto& [name, imageRelay] : _guiState._modelImageRelays) {
                bool isSelected = (_guiState._currentModelImage == name);
                if (ImGui::Selectable(name.c_str(), isSelected))
                    _guiState._currentModelImage = name;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (!_guiState._currentModelImage.empty())
            _guiState._modelImageRelays[_guiState._currentModelImage].render();

        ImGui::End(); // Training Images
    }
    else {
        ImGui::End(); // Training Images
    }

    imGuiRender();
}

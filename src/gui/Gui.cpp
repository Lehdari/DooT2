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
#include "gui/GameWindow.hpp"
#include "gui/PlotWindow.hpp"
#include "gui/ImagesWindow.hpp"
#include "Model.hpp"
#include "Trainer.hpp"

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

    // Add default windows (TODO replace with dynamic windowing)
    createWindow<gui::GameWindow>();
    createWindow<gui::PlotWindow>();
    createWindow<gui::ImagesWindow>();
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

    // Menu bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("New window")) {
            if (ImGui::MenuItem("Game window", nullptr))
                createWindow<gui::GameWindow>();
            if (ImGui::MenuItem("Plot window", nullptr))
                createWindow<gui::PlotWindow>();
            if (ImGui::MenuItem("Training images window", nullptr))
                createWindow<gui::ImagesWindow>();
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Make the entire window dockable
    ImGui::DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

    // Render windows
    for (auto& w : _windows) {
        if (!w)
            continue;

        w->render(trainer, model, &_guiState);

        // Check if window has been closed
        if (w->isClosed())
            w.reset();
    }

    imGuiRender();
}

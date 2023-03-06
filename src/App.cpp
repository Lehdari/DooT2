//
// Project: DooT2
// File: App.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "App.hpp"
#include "Constants.hpp"
#include "Trainer.hpp"
#include "Model.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "glad/glad.h"
#include "implot.h"

#include <chrono>


using namespace gvizdoom;


App::App(Trainer* trainer, Model* model) :
    _window     (nullptr),
    _glContext  (nullptr),
    _quit       (false),
    _trainer    (trainer),
    _model      (model)
{
    auto& doomGame = DoomGame::instance();

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Error: Could not initialize SDL!\n");
        return;
    }
    _window = SDL_CreateWindow(
        "DooT2",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        1920, // TODO settings
        1080, // TODO settings
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
    if (_window == nullptr) {
        printf("Error: SDL Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);

    _glContext = SDL_GL_CreateContext(_window);
    if (_glContext == nullptr) {
        printf("Error: SDL OpenGL context could not be created! SDL_Error: %s\n",
            SDL_GetError());
        return;
    }

    // Load OpenGL extensions
    if (!gladLoadGL()) {
        printf("Error: gladLoadGL failed\n");
        return;
    }

    // Initialize imgui
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForOpenGL(_window, _glContext);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImPlot::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Initialize OpenGL
    glViewport(0, 0, 1920, 1080); // TODO settings
    glClearColor(0.2f, 0.2f, 0.2f, 1.f);
    glEnable(GL_DEPTH_TEST);

    // Initialize GUI state
    _guiState._frameTexture.create(doomGame.getScreenWidth(), doomGame.getScreenHeight(),
        GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE);
    {   // Update the plot time series map
        auto timeSeriesHandle = _model->timeSeries.read();
        auto timeSeriesNames = timeSeriesHandle->getSeriesNames();
        for (const auto& name : timeSeriesNames) {
            if (name == "time") // "time" is dedicated to be used as plot x-coordinates when time mode is selected
                continue;
            auto* seriesVector = timeSeriesHandle->getSeriesVector<double>(name);
            if (seriesVector == nullptr) // should not happen, most likely caused by using wrong type (not double)
                throw std::runtime_error("No time series vector with name \"" + name + "\" found! (check the time series type)");
            _guiState._plotTimeSeriesVectors.emplace(name, std::make_pair(seriesVector, false));
        }
    }
    // Update the model images map
    for (auto& [name, imageBuffer] : _model->images) {
        _guiState._modelImageRelays.emplace(name, &imageBuffer);
    }
}

App::~App()
{
    // Destroy imgui
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    // Destroy window and quit SDL subsystems
    if (_glContext != nullptr)
        SDL_GL_DeleteContext(_glContext);
    if (_window != nullptr)
        SDL_DestroyWindow(_window);
    SDL_Quit();
}

void App::loop()
{
    using namespace std::chrono;

    constexpr double framerate = 60.0; // TODO settings

    SDL_Event event;
    while (!_quit) {
        auto frameBegin = high_resolution_clock::now();

        while(SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT ||
                (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) ||
                (event.type == SDL_KEYDOWN &&
                event.key.keysym.sym == SDLK_ESCAPE)) {
                _quit = true;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gui();

        // Introduce delay to cap the framerate
        auto frameEnd = high_resolution_clock::now();
        auto frameTime = duration_cast<microseconds>(frameEnd - frameBegin).count();
        int delayMs = std::max((int)std::floor((1000.0/framerate)-((double)frameTime*0.001)), 0);
        SDL_Delay(delayMs);

        // Swap draw and display buffers
        SDL_GL_SwapWindow(_window);
    }
}

void App::gui()
{
    auto& doomGame = DoomGame::instance();

    imGuiNewFrame();

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
        auto timeSeriesReadHandle = _model->timeSeries.read();
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
        auto timeSeriesReadHandle = _model->timeSeries.read();

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
            auto frameHandle = _trainer->getFrameReadHandle();
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

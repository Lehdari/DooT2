//
// Project: DooT2
// File: Gui.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ImageRelay.hpp"
#include "TimeSeries.hpp"

#include "gut_opengl/Texture.hpp"
#include "imgui.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"

#include <map>


class Trainer;
class Model;
struct SDL_Window;
typedef void *SDL_GLContext;


namespace gui {

class Gui {
public:
    ~Gui();

    void init(SDL_Window* window, SDL_GLContext* glContext);
    void update(Model* model);

    void render(SDL_Window* window, Trainer* trainer, Model* model);

private:
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

    State   _guiState;

    inline static void imGuiNewFrame(SDL_Window* window);
    inline static void imGuiRender();
};


void Gui::imGuiNewFrame(SDL_Window* window)
{
    // Initialize imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();
}

void Gui::imGuiRender()
{
    // Generate draw data
    ImGui::Render();

    // Render imgui
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace gui

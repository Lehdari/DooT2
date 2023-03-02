//
// Project: DooT2
// File: App.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <SDL.h>

#include "imgui.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"


class Trainer;
class Model;


class App {
public:
    App(Trainer* trainer, Model* model);
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*     _window;
    SDL_GLContext   _glContext;

    bool            _quit;

    Trainer*        _trainer;
    Model*          _model;

    inline void imGuiNewFrame() const;
    inline void imGuiRender() const;

    struct GuiState {
        bool    _lossPlotAutoFit    {false};
    };

    GuiState        _guiState;

    void gui();
};


void App::imGuiNewFrame() const
{
    // Initialize imgui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(_window);
    ImGui::NewFrame();
}

void App::imGuiRender() const
{
    // Generate draw data
    ImGui::Render();

    // Render imgui
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

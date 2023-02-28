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
#include "backends/imgui_impl_sdl.h"


class App {
public:
    App();
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*     _window;
    SDL_GLContext   _glContext;

    bool            _quit;

    inline void imGuiNewFrame() const;
    inline void imGuiRender() const;

    void gui() const;
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

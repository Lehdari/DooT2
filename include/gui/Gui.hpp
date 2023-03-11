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

#include "gui/Window.hpp"
#include "gui/ImageRelay.hpp"
#include "gui/State.hpp"
#include "TimeSeries.hpp"

#include "gut_opengl/Texture.hpp"
#include "imgui.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"

#include <map>
#include <set>


class Trainer;
class Model;
struct SDL_Window;
typedef void *SDL_GLContext;


namespace gui {

class Window;


class Gui {
public:
    ~Gui();

    void init(SDL_Window* window, SDL_GLContext* glContext);
    void update(Model* model);

    void render(SDL_Window* window, Trainer* trainer, Model* model);

private:
    State                                   _guiState;
    std::vector<std::unique_ptr<Window>>    _windows;

    inline static void imGuiNewFrame(SDL_Window* window);
    inline static void imGuiRender();

    // Create a new window of given type
    template <typename T_Window>
    void createWindow();

    // Get set of active ids of given window type
    template <typename T_Window>
    static std::set<int>* activeWindows();
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

template<typename T_Window>
void Gui::createWindow()
{
    // Find first available slot
    for (auto& w : _windows) {
        if (!w) {
            w.reset(new T_Window(activeWindows<T_Window>()));
            return;
        }
    }
    _windows.push_back(std::unique_ptr<gui::Window>(new T_Window(activeWindows<T_Window>())));
}

template<typename T_Window>
std::set<int>* Gui::activeWindows()
{
    static std::set<int> activeIds;
    return &activeIds;
}

} // namespace gui

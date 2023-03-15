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
#include "util/TimeSeries.hpp"

#include "gut_opengl/Texture.hpp"
#include "imgui.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"

#include <map>
#include <set>


namespace ml {

class Trainer;
class Model;

} // namespace ml

struct SDL_Window;
typedef void *SDL_GLContext;


namespace gui {

class Window;


class Gui {
public:
    ~Gui();

    void init(SDL_Window* window, SDL_GLContext* glContext);
    void update(ml::Model* model);
    void createDefaultLayout();
    void loadLayout(const std::filesystem::path& layoutFilename);
    void saveLayout(const std::filesystem::path& layoutFilename) const;

    void render(SDL_Window* window, ml::Trainer* trainer, ml::Model* model);

private:
    State                                   _guiState;
    std::vector<std::unique_ptr<Window>>    _windows;

    inline static void imGuiNewFrame(SDL_Window* window);
    inline static void imGuiRender();

    // Create a new window of given type
    template <typename T_Window>
    void createWindow(int id = -1, const nlohmann::json* config = nullptr);

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
void Gui::createWindow(int id, const nlohmann::json* config)
{
    // Explicit ID requested, try to create a window with the ID
    if (id >= 0) {
        if (activeWindows<T_Window>()->contains(id)) // ID already exists, do nothing
            return;
        _windows.push_back(std::unique_ptr<gui::Window>(new T_Window(activeWindows<T_Window>(), id)));
        _windows.back()->update(&_guiState);
        if (config != nullptr)
            _windows.back()->applyConfig(*config);
        return;
    }

    // Try to find first available slot (nullptr) in the _windows vector and reuse it
    for (auto& w : _windows) {
        if (!w) {
            w.reset(new T_Window(activeWindows<T_Window>()));
            w->update(&_guiState);
            if (config != nullptr)
                w->applyConfig(*config);
            return;
        }
    }

    // No free slots, create a completely new window
    _windows.push_back(std::unique_ptr<gui::Window>(new T_Window(activeWindows<T_Window>())));
    _windows.back()->update(&_guiState);
    if (config != nullptr)
        _windows.back()->applyConfig(*config);
}

template<typename T_Window>
std::set<int>* Gui::activeWindows()
{
    static std::set<int> activeIds;
    return &activeIds;
}

} // namespace gui

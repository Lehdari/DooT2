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
#include "gui/Windows.hpp"
#include "gui/WindowTypeUtils.hpp"
#include "ml/Model.hpp"
#include "ml/TrainingInfo.hpp"
#include "ml/Trainer.hpp"

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

    _guiState.frameTexture.create(doomGame.getScreenWidth(), doomGame.getScreenHeight(),
        GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE);

    // Initialize imgui
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForOpenGL(window, glContext);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImPlot::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

void Gui::update(ml::TrainingInfo* trainingInfo)
{
    _guiState.timeSeries["training"] = &trainingInfo->trainingTimeSeries;
    _guiState.timeSeries["evaluation"] = &trainingInfo->evaluationTimeSeries;

    // Update the model image relays map
    _guiState.modelImageRelays.clear();
    for (auto& [name, imageBuffer] : trainingInfo->images) {
        _guiState.modelImageRelays.emplace(name, &imageBuffer);
    }

    // Update all the windows
    for (auto& w : _windows) {
        w->update();
    }
}

void Gui::createDefaultLayout()
{
    _windows.clear();
    createWindow<gui::GameWindow>();
    createWindow<gui::PlotWindow>();
    createWindow<gui::ImagesWindow>();
}

void Gui::loadLayout(const std::filesystem::path& layoutFilename, SDL_Window* window)
{
    std::ifstream layoutFile(layoutFilename);
    auto layout = nlohmann::json::parse(layoutFile);

    // Create and initialize windows from configs
    auto windowLayouts = layout["windows"].get<std::vector<nlohmann::json>>();
    for (auto& w : windowLayouts) {
        auto* config = &w["config"];
        if (config->is_null())
            config = nullptr;

        // Create a window: the lambda callback fetches the correct window type
        windowTypeNameCallback(w["type"].get<std::string>(), [&]<typename T_Window>() {
            createWindow<T_Window>(w["id"], config);
        });
    }

    // Set window dimensions and position
    if (window != nullptr) {
        if (layout.contains("windowWidth") && layout.contains("windowHeight")) {
            int w = layout["windowWidth"].get<int>();
            int h = layout["windowHeight"].get<int>();
            SDL_SetWindowSize(window, w, h);
            glViewport(0, 0, w, h);
        }
        if (layout.contains("windowPosX") && layout.contains("windowPosY")) {
            int x = layout["windowPosX"].get<int>();
            int y = layout["windowPosY"].get<int>();
            SDL_SetWindowPosition(window, x, y);
        }
    }
}

void Gui::saveLayout(const std::filesystem::path& layoutFilename, SDL_Window* window) const
{
    nlohmann::json layout;
    if (window != nullptr) {
        int x, y, w, h;
        int t, l, b, r;
        SDL_GetWindowBordersSize(window, &t, &l, &b, &r);
        SDL_GetWindowPosition(window, &x, &y);
        SDL_GetWindowSize(window, &w, &h);
        layout["windowWidth"] = w;
        layout["windowHeight"] = h;
        layout["windowPosX"] = x;
        layout["windowPosY"] = y-t;
    }

    std::vector<nlohmann::json> windowLayouts;
    for (auto& w : _windows) {
        if (!w)
            continue;
        windowLayouts.emplace_back();
        windowLayouts.back()["type"] = windowTypeName(w->getTypeId());;
        windowLayouts.back()["id"] = w->getId();
        windowLayouts.back()["config"] = w->getConfig();
    }

    layout["windows"] = windowLayouts;
    std::ofstream layoutFile(layoutFilename);
    layoutFile << std::setw(4) << layout << std::endl;
}

void Gui::render(SDL_Window* window, ml::Trainer* trainer)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    imGuiNewFrame(window);

    // Menu bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("New window")) {
            windowForEachTypeCallback([&]<typename T_Window>() {
                if (ImGui::MenuItem(WindowTypeInfo<T_Window>::label, nullptr))
                    createWindow<T_Window>();
            });
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

        w->render(trainer);

        // Check if window has been closed
        if (w->isClosed())
            w.reset();
    }

    imGuiRender();
}

void Gui::setCallback(const std::string& name, std::function<void(const State&)>&& callback)
{
    _guiState.callbacks[name] = callback;
}

const State& Gui::getState() const noexcept
{
    return _guiState;
}

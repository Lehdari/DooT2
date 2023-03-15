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
#include "gui/WindowTypeUtils.hpp"
#include "gui/GameWindow.hpp"
#include "gui/ImagesWindow.hpp"
#include "gui/PlotWindow.hpp"
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
}

void Gui::update(Model* model)
{
    _guiState._timeSeries["training"] = &model->timeSeries;

    // Update the model image relays map
    _guiState._modelImageRelays.clear();
    for (auto& [name, imageBuffer] : model->images) {
        _guiState._modelImageRelays.emplace(name, &imageBuffer);
    }

    // Update all the windows
    for (auto& w : _windows) {
        w->update(&_guiState);
    }
}

void Gui::createDefaultLayout()
{
    _windows.clear();
    createWindow<gui::GameWindow>();
    createWindow<gui::PlotWindow>();
    createWindow<gui::ImagesWindow>();
}

void Gui::loadLayout(const std::filesystem::path& layoutFilename)
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
}

void Gui::saveLayout(const std::filesystem::path& layoutFilename) const
{
    std::vector<nlohmann::json> windowLayouts;
    for (auto& w : _windows) {
        windowLayouts.emplace_back();
        windowLayouts.back()["type"] = windowTypeName(w->getTypeId());;
        windowLayouts.back()["id"] = w->getId();
        windowLayouts.back()["config"] = w->getConfig();
    }

    nlohmann::json layout;
    layout["windows"] = windowLayouts;
    std::ofstream layoutFile(layoutFilename);
    layoutFile << std::setw(4) << layout << std::endl;
}

void Gui::render(SDL_Window* window, Trainer* trainer, Model* model)
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

        w->render(trainer, model, &_guiState);

        // Check if window has been closed
        if (w->isClosed())
            w.reset();
    }

    imGuiRender();
}

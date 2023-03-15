//
// Project: DooT2
// File: GameWindow.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/GameWindow.hpp"
#include "gui/State.hpp"
#include "ml/Trainer.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "imgui.h"


using namespace ml;


void gui::GameWindow::update(gui::State* guiState)
{
}

void gui::GameWindow::render(ml::Trainer* trainer, ml::Model* model, gui::State* guiState)
{
    if (!_open) return;

    auto& doomGame = gvizdoom::DoomGame::instance();

    if (ImGui::Begin(("Game " + std::to_string(_id)).c_str(), &_open)) {
        {
            auto frameHandle = trainer->getFrameReadHandle();
            guiState->_frameTexture.updateFromBuffer(frameHandle->data(), GL_BGRA);
        }
        ImGui::Image((void*)(intptr_t)guiState->_frameTexture.id(),
            ImVec2(doomGame.getScreenWidth(), doomGame.getScreenHeight()),
            ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
        ImGui::End();
    }
    else
        ImGui::End();
}

void gui::GameWindow::applyConfig(const nlohmann::json& config)
{
}

nlohmann::json gui::GameWindow::getConfig() const
{
    return {};
}

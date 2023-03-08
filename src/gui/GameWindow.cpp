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
#include "Trainer.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "imgui.h"


void gui::GameWindow::render(Trainer* trainer, Model* model, gui::State* guiState) const
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    if (guiState->_showFrame && ImGui::Begin("Game", &guiState->_showFrame)) {
        {
            auto frameHandle = trainer->getFrameReadHandle();
            guiState->_frameTexture.updateFromBuffer(frameHandle->data(), GL_BGRA);
        }
        ImGui::Image((void*)(intptr_t)guiState->_frameTexture.id(),
            ImVec2(doomGame.getScreenWidth(), doomGame.getScreenHeight()),
            ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));
        ImGui::End();
    }
    else {
        ImGui::End();
    }
}

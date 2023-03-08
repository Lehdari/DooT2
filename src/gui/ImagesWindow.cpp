//
// Project: DooT2
// File: ImagesWindow.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/ImagesWindow.hpp"
#include "gui/State.hpp"

#include "imgui.h"


void gui::ImagesWindow::render(Trainer* trainer, Model* model, gui::State* guiState) const
{
    if (guiState->_showTrainingImages && ImGui::Begin("Training Images", &guiState->_showTrainingImages)) {
        if (ImGui::BeginCombo("##combo", guiState->_currentModelImage.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto& [name, imageRelay] : guiState->_modelImageRelays) {
                bool isSelected = (guiState->_currentModelImage == name);
                if (ImGui::Selectable(name.c_str(), isSelected))
                    guiState->_currentModelImage = name;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (!guiState->_currentModelImage.empty())
            guiState->_modelImageRelays[guiState->_currentModelImage].render();

        ImGui::End(); // Training Images
    }
    else {
        ImGui::End(); // Training Images
    }
}

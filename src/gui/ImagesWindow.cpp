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


gui::ImagesWindow::ImagesWindow(std::set<int>* activeIds) :
    Window              (activeIds),
    _currentModelImage  ("")
{
}

void gui::ImagesWindow::update(gui::State* guiState)
{
}

void gui::ImagesWindow::render(Trainer* trainer, Model* model, gui::State* guiState)
{
    if (!_open) return;
    if (ImGui::Begin(("Images " + std::to_string(_id)).c_str(), &_open)) {
        if (ImGui::BeginCombo("##combo", _currentModelImage.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto& [name, imageRelay] : guiState->_modelImageRelays) {
                bool isSelected = (_currentModelImage == name);
                if (ImGui::Selectable(name.c_str(), isSelected))
                    _currentModelImage = name;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (!_currentModelImage.empty())
            guiState->_modelImageRelays[_currentModelImage].render();

        ImGui::End();
    }
    else
        ImGui::End();
}

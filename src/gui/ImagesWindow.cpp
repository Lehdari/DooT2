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


gui::ImagesWindow::ImagesWindow(std::set<int>* activeIds, State* guiState, int id) :
    Window(this, guiState, activeIds, id),
    _activeImage    ("")
{
}

void gui::ImagesWindow::update()
{
}

void gui::ImagesWindow::render(ml::Trainer* trainer)
{
    if (!_open) return;
    if (ImGui::Begin(("Images " + std::to_string(_id)).c_str(), &_open)) {
        if (ImGui::BeginCombo("##combo", _activeImage.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto& [name, imageRelay] : _guiState->modelImageRelays) {
                bool isSelected = (_activeImage == name);
                if (ImGui::Selectable(name.c_str(), isSelected))
                    _activeImage = name;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (!_activeImage.empty() && _guiState->modelImageRelays.contains(_activeImage))
            _guiState->modelImageRelays[_activeImage].render();

        ImGui::End();
    }
    else
        ImGui::End();
}

void gui::ImagesWindow::applyConfig(const nlohmann::json& config)
{
    _activeImage = config["activeImage"].get<std::string>();
}

nlohmann::json gui::ImagesWindow::getConfig() const
{
    nlohmann::json config;
    config["activeImage"] = _activeImage;
    return config;
}

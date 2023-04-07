//
// Project: DooT2
// File: TrainingWindow.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/TrainingWindow.hpp"
#include "gui/State.hpp"

#include "imgui.h"


gui::TrainingWindow::TrainingWindow(std::set<int>* activeIds, State* guiState, int id) :
    Window(this, guiState, activeIds, id)
{
}

void gui::TrainingWindow::update()
{
}

void gui::TrainingWindow::render(ml::Trainer* trainer)
{
    if (!_open) return;
    if (ImGui::Begin(("Training " + std::to_string(_id)).c_str(), &_open)) {
        float fontSize = ImGui::GetFontSize();

        // TODO model select menus

        // Training controls
        ImGui::Text("Training:");

        // Start / Pause / Continue button
        float startButtonWidth = fontSize*5.0f;
        ImGui::SameLine();
        if (_guiState->trainingStatus == State::TrainingStatus::STOPPED) {
            if (ImGui::Button("Start", ImVec2(startButtonWidth, 0.0f))) {
                printf("Starting training\n");
                _guiState->trainingStatus = State::TrainingStatus::ONGOING;
            }
        }
        else {
            if (_guiState->trainingStatus == State::TrainingStatus::ONGOING) {
                if (ImGui::Button("Pause", ImVec2(startButtonWidth, 0.0f))) {
                    printf("Pausing training\n");
                    _guiState->trainingStatus = State::TrainingStatus::PAUSED;
                }
            }
            else { // PAUSED
                if (ImGui::Button("Continue", ImVec2(startButtonWidth, 0.0f))) {
                    printf("Continuing training\n");
                    _guiState->trainingStatus = State::TrainingStatus::ONGOING;
                }
            }
        }

        // Stop button
        float stopButtonWidth = fontSize*3.0f;
        ImGui::SameLine();
        ImGui::BeginDisabled(_guiState->trainingStatus == State::TrainingStatus::STOPPED); // disable if there's no training process to stop
        if (ImGui::Button("Stop", ImVec2(stopButtonWidth, 0.0f))) {
            printf("Stopping training\n");
            _guiState->trainingStatus = State::TrainingStatus::STOPPED;
        }
        ImGui::EndDisabled();

        ImGui::End();
    }
    else
        ImGui::End();
}

void gui::TrainingWindow::applyConfig(const nlohmann::json& config)
{
}

nlohmann::json gui::TrainingWindow::getConfig() const
{
    nlohmann::json config;
    return config;
}

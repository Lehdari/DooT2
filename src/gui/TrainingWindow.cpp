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
#include "ml/Models.hpp"
#include "ml/ModelTypeUtils.hpp"
#include "ml/Trainer.hpp"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"


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
        ImVec2 windowSize = ImGui::GetWindowSize();

        // Flag for disabling all settings when training's in progress
        bool trainingInProgress = _guiState->trainingStatus != State::TrainingStatus::STOPPED;
        ImGui::BeginDisabled(trainingInProgress);

        // Experiment name input
        ImGui::Text("Experiment name:");
        ImGui::SetNextItemWidth(windowSize.x);
        if (ImGui::InputText("##ExperimentName", _guiState->experimentName, 255)) {
            if (_guiState->callbacks.contains("newModelTypeSelected"))
                _guiState->callbacks["newModelTypeSelected"](*_guiState);
        }

        // Model select
        ImGui::Text("Model:   ");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(windowSize.x - fontSize*6.35f);
        if (ImGui::BeginCombo("##ModelSelector", _guiState->modelTypeName.c_str())) {
            ml::modelForEachTypeCallback([&]<typename T_Model>() {
                constexpr auto name = ml::ModelTypeInfo<T_Model>::name;
                bool isSelected = (_guiState->modelTypeName == name);
                if (ImGui::Selectable(name, isSelected)) {
                    // Call the callback function for new model type selection (in case it's defined)
                    if (_guiState->modelTypeName != name && _guiState->callbacks.contains("newModelTypeSelected")) {
                        _guiState->modelTypeName = name;
                        _guiState->callbacks["newModelTypeSelected"](*_guiState);
                    }
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            });

            ImGui::EndCombo();
        }

        ImGui::EndDisabled();

        // Model config table
        if (ImGui::CollapsingHeader("Model configuration")) {
            ImGui::BeginDisabled(trainingInProgress);

            auto* experimentConfig = trainer->getExperimentConfig();
            assert(experimentConfig != nullptr);
            if (experimentConfig->contains("model_config")) {
                auto& modelConfig = (*experimentConfig)["model_config"];
                if (ImGui::BeginTable("Model configuration", 2,
                    ImGuiTableFlags_Borders |
                    ImGuiTableFlags_SizingFixedFit |
                    ImGuiTableFlags_SizingFixedSame)) {
                    int id = 0;
                    // List all config entries
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0,0});
                    for (auto& [paramName, paramValue]: modelConfig.items()) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%s", paramName.c_str());
                        ImGui::TableSetColumnIndex(1);
                        float columnWidth = ImGui::GetColumnWidth(1);
                        ImGui::PushID(id++);
                        switch (paramValue.type()) {
                            case nlohmann::json::value_t::boolean: {
                                bool v = paramValue.get<bool>();
                                ImGui::Checkbox("##value", &v);
                                paramValue = v;
                            }   break;
                            case nlohmann::json::value_t::number_integer: {
                                int v = paramValue.get<int>();
                                ImGui::SetNextItemWidth(columnWidth);
                                ImGui::InputInt("##value", &v);
                                paramValue = v;
                            }   break;
                            case nlohmann::json::value_t::number_float: {
                                double v = paramValue.get<double>();
                                ImGui::SetNextItemWidth(columnWidth);
                                ImGui::InputDouble("##value", &v, 0.0, 0.0, "%.5g");
                                paramValue = v;
                            }   break;
                            case nlohmann::json::value_t::string: {
                                std::string v = paramValue.get<std::string>();
                                ImGui::SetNextItemWidth(columnWidth);
                                if (ImGui::InputText("##value", &v))
                                    paramValue = v;
                            }   break;
                            default: {
                                ImGui::Text("Unsupported type");
                            }   break;
                        }
                        ImGui::PopID();
                    }
                    ImGui::PopStyleVar();
                    ImGui::EndTable();
                }
            }

            ImGui::EndDisabled();
        }

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
    if (config.contains("modelTypeName"))
        _guiState->modelTypeName = config["modelTypeName"].get<std::string>();
}

nlohmann::json gui::TrainingWindow::getConfig() const
{
    nlohmann::json config;
    config["modelTypeName"] = _guiState->modelTypeName;
    return config;
}

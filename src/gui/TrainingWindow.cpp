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
#include "util/ExperimentUtils.hpp"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"


namespace fs = std::filesystem;


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
    int id = 0;
    auto* experimentConfig = trainer->getExperimentConfig();

    if (ImGui::Begin(("Training " + std::to_string(_id)).c_str(), &_open)) {
        float fontSize = ImGui::GetFontSize();
        ImVec2 windowSize = ImGui::GetWindowSize();

        // Flag for disabling all settings when training's in progress
        bool trainingInProgress = _guiState->trainingStatus != State::TrainingStatus::STOPPED;

        // Experiment configuration
        if (ImGui::CollapsingHeader("Experiment configuration")) {
            ImGui::BeginDisabled(trainingInProgress);

            // Experiment name input
            ImGui::Text("Experiment name:");
            ImGui::SetNextItemWidth(windowSize.x - fontSize * 2.0f);
            ImGui::InputText("##ExperimentName", &_guiState->experimentName);

            // Experiment base input
            ImGui::Text("Experiment base:");
            ImGui::SetNextItemWidth(windowSize.x - fontSize * 2.0f);
            if (ImGui::InputText("##ExperimentBase", &_guiState->experimentBase) &&
                !_guiState->experimentBase.empty()) {
                auto baseRoot = experimentRootFromString(_guiState->experimentBase);
                if (fs::exists(baseRoot)) { // valid experiment root passed, let's check if there's a config
                    fs::path baseExperimentConfigFilename = baseRoot / "experiment_config.json";
                    if (fs::exists(baseExperimentConfigFilename)) { // config found, parse it
                        std::ifstream baseExperimentConfigFile(baseExperimentConfigFilename);
                        _guiState->baseExperimentConfig = nlohmann::json::parse(baseExperimentConfigFile);
                        if (_guiState->baseExperimentConfig.contains("model_type")) { // reset the model
                            _guiState->modelTypeName = _guiState->baseExperimentConfig["model_type"];
                            if (_guiState->callbacks.contains("resetExperiment"))
                                _guiState->callbacks["resetExperiment"](*_guiState);
                        }
                        else
                            printf("WARNING: No model_type specified in the base experiment config\n"); // TODO logging
                    }
                    else
                        printf("WARNING: No experiment config from %s found!\n", baseExperimentConfigFilename.c_str());
                }
            }
            if (ImGui::Button("Reload model config")) { // reload model config from the base experiment
                if (_guiState->baseExperimentConfig.contains("model_config"))
                    (*trainer->getExperimentConfig())["model_config"] = _guiState->baseExperimentConfig["model_config"];
                else
                    printf("WARNING: No base experiment model config found\n"); // TODO logging
            }

            // Training task options
            ImGui::BeginDisabled(!_guiState->experimentBase.empty()); // Same model type forced when using a base experiment
            ImGui::Text("Training task:");
            static const std::string taskNames[] = {"Frame Encoding", "Agent Policy"};
            ImGui::SetNextItemWidth(windowSize.x - fontSize * 2.0f);
            if (ImGui::BeginCombo("##TaskSelector", taskNames[(int32_t)_guiState->trainingTask].c_str())) {
                for (int i=0; i<2; ++i) {
                    auto name = taskNames[i].c_str();
                    bool isSelected = (int32_t)_guiState->trainingTask == i;
                    if (ImGui::Selectable(name, isSelected)) {
                        _guiState->trainingTask = (State::TrainingTask)i;
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                };

                ImGui::EndCombo();
            }

            ImGui::Text("Model to train:");
            ImGui::SetNextItemWidth(windowSize.x - fontSize * 2.0f);
            if (ImGui::BeginCombo("##ModelSelector", _guiState->modelTypeName.c_str())) {
                ml::modelForEachTypeCallback([&]<typename T_Model>() {
                    constexpr auto name = ml::ModelTypeInfo<T_Model>::name;
                    bool isSelected = (_guiState->modelTypeName == name);
                    if (ImGui::Selectable(name, isSelected)) {
                        // Call the callback function for new model type selection (in case it's defined)
                        if (_guiState->modelTypeName != name && _guiState->callbacks.contains("resetExperiment")) {
                            _guiState->modelTypeName = name;
                            _guiState->callbacks["resetExperiment"](*_guiState);
                        }
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                });

                ImGui::EndCombo();
            }
            ImGui::EndDisabled();

            ImGui::EndDisabled();
        }

        // Model configuration
        if (ImGui::CollapsingHeader("Model configuration")) {
            ImGui::BeginDisabled(trainingInProgress);

            assert(experimentConfig != nullptr);
            if (experimentConfig->contains("model_config")) {
                auto& modelConfig = (*experimentConfig)["model_config"];
                if (ImGui::BeginTable("Model configuration", 2,
                    ImGuiTableFlags_Borders |
                    ImGuiTableFlags_SizingFixedFit |
                    ImGuiTableFlags_SizingFixedSame)) {
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
                            case nlohmann::json::value_t::number_integer:
                            case nlohmann::json::value_t::number_unsigned: {
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

        // Grid search setup
        if (ImGui::CollapsingHeader("Grid search")) {
            ImGui::BeginDisabled(trainingInProgress);
            ImGui::Checkbox("Perform a grid search", &_guiState->gridSearch);

            ImGui::BeginDisabled(!_guiState->gridSearch);

            auto& gridSearchConfig = _guiState->gridSearchModelConfigParams;
            int maxNParamValues = 0;
            for (auto& [paramName, paramValues] : gridSearchConfig.items()) {
                if (paramValues.size() > maxNParamValues) {
                    maxNParamValues = paramValues.size();
                }
            }

            if (experimentConfig->contains("model_config")) {
                auto& modelConfig = (*experimentConfig)["model_config"];
                if (ImGui::BeginTable("Grid search configuration", maxNParamValues+2,
                    ImGuiTableFlags_Borders |
                    ImGuiTableFlags_ScrollX |
                    ImGuiTableFlags_Resizable)) {
                    // List all config entries
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {0,0});
                    for (auto& [paramName, paramValues]: gridSearchConfig.items()) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);

                        // First column: the parameter selection
                        ImGui::SetNextItemWidth(ImGui::GetColumnWidth()-fontSize*1.8f);
                        ImGui::PushID(id++);
                        bool paramChanged = false;
                        if (ImGui::BeginCombo("##gridSearchParamName", paramName.c_str())) {
                            for (auto& [pName, pValue] : modelConfig.items()) {
                                bool isSelected = (pName == paramName);
                                if (!gridSearchConfig.contains(pName) &&
                                    ImGui::Selectable(pName.c_str(), isSelected)) {
                                    gridSearchConfig.erase(paramName);
                                    gridSearchConfig[pName] = { pValue };
                                    paramChanged = true;
                                }
                            }
                            ImGui::EndCombo();
                        }
                        // Parameter removal button
                        ImGui::SameLine();
                        if (ImGui::Button("X##removeGridSearchParam")) {
                            gridSearchConfig.erase(paramName);
                            paramChanged = true;
                        }
                        // Tooltip for the removal button
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                            ImGui::SetTooltip("Remove the parameter");
                        ImGui::PopID();
                        if (paramChanged)
                            break;

                        // Find parameter type
                        auto paramType = nlohmann::json::value_t::null; // null means parameter not found in model config
                        if (modelConfig.contains(paramName))
                            paramType = modelConfig[paramName].type();

                        int columnId = 0;
                        for (auto& paramValue : paramValues) {
                            ImGui::TableSetColumnIndex(++columnId);
                            float columnWidth = ImGui::GetColumnWidth();
                            ImGui::PushID(id++);
                            switch (paramType) {
                                case nlohmann::json::value_t::boolean: {
                                    bool v = paramValue.get<bool>();
                                    ImGui::Checkbox("##value", &v);
                                    paramValue = v;
                                }   break;
                                case nlohmann::json::value_t::number_integer:
                                case nlohmann::json::value_t::number_unsigned: {
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
                                case nlohmann::json::value_t::null: {
                                    ImGui::Text("No such model parameter");
                                }   break;
                                default: {
                                    ImGui::Text("Unsupported type");
                                }   break;
                            }
                            ImGui::PopID();
                        }

                        // Last column for adding values
                        ImGui::TableSetColumnIndex(++columnId);
                        ImGui::PushID(id++);
                        if (ImGui::Button("Add")) {
                            gridSearchConfig[paramName].push_back(paramValues.back());
                        }
                        ImGui::PopID();
                    }

                    // Last row for adding parameters
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    {
                        ImGui::PushID(id++);
                        if (ImGui::BeginCombo("##gridSearchParamName", "Add parameter")) {
                            for (auto& [pName, pValue] : modelConfig.items()) {
                                if (!gridSearchConfig.contains(pName) &&
                                    ImGui::Selectable(pName.c_str())) {
                                    gridSearchConfig[pName] = { pValue };
                                }
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::PopID();
                    }

                    ImGui::PopStyleVar();
                    ImGui::EndTable();
                }
            }

            ImGui::EndDisabled();
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

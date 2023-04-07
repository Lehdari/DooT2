//
// Project: DooT2
// File: TrainingWindow.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "gui/Window.hpp"


namespace gui {

class TrainingWindow : public Window {
public:
    TrainingWindow(std::set<int>* activeIds, State* guiState, int id = -1);

    void update() override;
    void render(ml::Trainer* trainer) override;
    void applyConfig(const nlohmann::json& config) override;
    nlohmann::json getConfig() const override;
};

};

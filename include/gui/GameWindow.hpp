//
// Project: DooT2
// File: GameWindow.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Window.hpp"


namespace gui {

class GameWindow : public Window {
public:
    GameWindow(std::set<int>* activeIds, State* guiState, int id = -1) :
        Window(this, guiState, activeIds, id)
    {}

    void update() override;
    void render(ml::Trainer* trainer) override;
    void applyConfig(const nlohmann::json& config) override;
    nlohmann::json getConfig() const override;
};

};

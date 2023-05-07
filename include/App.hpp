//
// Project: DooT2
// File: App.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "gui/Gui.hpp"

#include <SDL.h>

#include <thread>


namespace gui {

class State;

} // namespace gui

namespace ml {

class Trainer;
class Model;

} // namespace ml


class App {
public:
    App(ml::Trainer* trainer);
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*                 _window;
    SDL_GLContext               _glContext;
    bool                        _quit;
    ml::Trainer*                _trainer;
    std::thread                 _trainerThread;
    gui::Gui                    _gui;
    std::vector<nlohmann::json> _gridSearchParameters; // flattened list of parameters for grid search
    size_t                      _gridSearchId; // ID of the current grid search experiment


    void trainingControl();

    // For updating possible changes made in GUI
    void updateExperimentConfig(nlohmann::json& experimentConfig);

    // Reset the whole experiment, create new config and instantiate new model.
    // Shouldn't be called when training thread is running.
    void resetExperiment();
};




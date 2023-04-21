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

    void trainingControl();
    void resetExperiment(); // shouldn't be called when training thread is running
};




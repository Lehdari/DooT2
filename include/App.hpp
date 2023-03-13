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


class Trainer;
class Model;


class App {
public:
    App(Trainer* trainer, Model* model);
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*     _window;
    SDL_GLContext   _glContext;

    bool            _quit;

    Trainer*        _trainer;
    Model*          _model;

    gui::Gui        _gui;
};




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


namespace ml {

class Trainer;

} // namespace ml


class App {
public:
    App(ml::Trainer* trainer);
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*     _window;
    SDL_GLContext   _glContext;

    bool            _quit;

    ml::Trainer*    _trainer;

    gui::Gui        _gui;
};




//
// Project: DooT2
// File: Window.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


class Trainer;
class Model;


namespace gui {

class State;


class Window {
public:
    virtual ~Window() = default;

    virtual void render(Trainer* trainer, Model* model, gui::State* guiState) const = 0;
};

} // namespace gui

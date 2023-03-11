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

#include <set>


class Trainer;
class Model;


namespace gui {

class State;


class Window {
public:
    Window(std::set<int>* activeIds);
    virtual ~Window();

    virtual void render(Trainer* trainer, Model* model, gui::State* guiState) = 0;

    bool isClosed() const noexcept;

protected:
    std::set<int>*  _activeIds;
    int             _id;
    bool            _open   {true}; // set to false to close the window

    int findFreeId();
};

} // namespace gui

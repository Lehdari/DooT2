//
// Project: DooT2
// File: Window.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "gui/Window.hpp"

#include <stdexcept>


int gui::Window::_nTypeIds = 0;


gui::Window::~Window()
{
    _activeIds->erase(_id);
}

bool gui::Window::isClosed() const noexcept
{
    return !_open;
}

int gui::Window::getId() const noexcept
{
    return _id;
}

int gui::Window::getTypeId() const noexcept
{
    return _typeId;
}

int gui::Window::findFreeId()
{
    // Try to find a free id
    for (int id=0; id<_activeIds->size(); ++id) {
        if (!_activeIds->contains(id))
            return id;
    }
    return (int)_activeIds->size();
}

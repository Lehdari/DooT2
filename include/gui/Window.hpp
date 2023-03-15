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

#include "WindowTypeUtils.hpp"

#include "nlohmann/json.hpp"

#include <set>


class Trainer;
class Model;


namespace gui {

class State;


// When implementing a new class derived from Window, remember to list it in the macro
// in the beginning of gui/WindowTypeUtils.hpp
class Window {
public:
    // By providing id >= 0, the constructor will try to create a window with an explicit ID.
    // This will fail and throw an exception in case the ID is already in use.
    // When id < 0, an unique ID will be dynamically assigned.
    template <typename T_Window>
    Window(T_Window* window, std::set<int>* activeIds, int id = -1);
    virtual ~Window();

    // Called when there's changes in gui state that might require synchronization between
    // it and the window's internal state
    virtual void update(gui::State* guiState) = 0;

    // Called every frame in intent to render the window to screen
    virtual void render(Trainer* trainer, Model* model, gui::State* guiState) = 0;

    // Apply window state defined in a configuration JSON object
    virtual void applyConfig(const nlohmann::json& config) = 0;

    // Extract window state into a configuration JSON object
    virtual nlohmann::json getConfig() const = 0;

    bool isClosed() const noexcept;
    int getId() const noexcept;
    int getTypeId() const noexcept;

protected:
    std::set<int>*  _activeIds;
    int             _id;
    int             _typeId;
    bool            _open   {true}; // set to false to close the window

    int findFreeId();
};


template <typename T_Window>
Window::Window(T_Window* window, std::set<int>* activeIds, int id) :
    _activeIds  (activeIds),
    _id         (id < 0 ? findFreeId() : id),
    _typeId     (WindowTypeInfo<T_Window>::id)
{
    if (_activeIds->contains(id))
        throw std::runtime_error("Window id " + std::to_string(id) + " already in use");
    _activeIds->emplace(_id);
}

} // namespace gui

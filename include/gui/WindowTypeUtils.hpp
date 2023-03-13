//
// Project: DooT2
// File: WindowTypeUtils.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "gui/Window.hpp"
#include "gui/GameWindow.hpp"
#include "gui/ImagesWindow.hpp"
#include "gui/PlotWindow.hpp"

#include <type_traits>

// List all available window types in this macro
#define GUI_WINDOW_TYPES            \
    GUI_WINDOW_TYPE(GameWindow)     \
    GUI_WINDOW_TYPE(ImagesWindow)   \
    GUI_WINDOW_TYPE(PlotWindow)


namespace gui {

template <typename T_Window, typename T>
using EnableIfWindowType = typename std::enable_if_t<std::is_base_of_v<Window, T_Window>, T>;


// In case you get "use of deleted function" error from the following function template,
// you most probably forgot to add it to the GUI_WINDOW_TYPES macro in the beginning of
// the file (remember to include the header too :))
template <typename T_Window>
constexpr EnableIfWindowType<T_Window, const char*> windowTypeName() = delete;

#define GUI_WINDOW_TYPE(WINDOW)                \
template <>                                    \
constexpr const char* windowTypeName<WINDOW>() \
{                                              \
    return #WINDOW;                            \
}
GUI_WINDOW_TYPES
#undef GUI_WINDOW_TYPE

std::string windowTypeName(int typeId)
{
    #define GUI_WINDOW_TYPE(WINDOW) \
    if (typeId == Window::typeId<WINDOW>()) \
        return windowTypeName<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
}

// Callbacks for fetching window type using type name or ID,
// intended to be used with a generic (templated) lambda:
// windowTypeCallback(windowTypeName, []<typename T>(){ /* T is the window type here */ });
template <typename F>
void windowTypeCallback(const std::string& typeName, F&& f)
{
    #define GUI_WINDOW_TYPE(WINDOW) \
    if (typeName == windowTypeName<WINDOW>()) \
        return f.template operator()<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
}

template <typename F>
void windowTypeCallback(int typeId, F&& f)
{
    #define GUI_WINDOW_TYPE(WINDOW) \
    if (typeId == Window::typeId<WINDOW>()) \
        return f.template operator()<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
}

} // namespace gui

#undef GUI_WINDOW_TYPES
#undef GUI_ADD_WINDOW_TYPE_NAME

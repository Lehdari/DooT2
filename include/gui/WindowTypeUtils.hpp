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

#include <type_traits>
#include <string>
#include <stdexcept>

#include "util/TypeCounter.hpp"


// List all available window types in this macro
// First argument: window type
// Second argument: window label in the "New window" menu
#define GUI_WINDOW_TYPES                          \
    GUI_WINDOW_TYPE(GameWindow,     "Game"      ) \
    GUI_WINDOW_TYPE(ImagesWindow,   "Images"    ) \
    GUI_WINDOW_TYPE(PlotWindow,     "Plotting"  ) \
    GUI_WINDOW_TYPE(TrainingWindow, "Training"  )

namespace gui {

#define GUI_WINDOW_TYPE(WINDOW, LABEL) class WINDOW;
GUI_WINDOW_TYPES
#undef GUI_WINDOW_TYPE

} // namespace gui

namespace detail {

// Generate the type counter for IDs
#define GUI_WINDOW_TYPE(WINDOW, LABEL) gui::WINDOW,
using WindowTypeCounter = TypeCounter<GUI_WINDOW_TYPES void>;
#undef GUI_WINDOW_TYPE

} // namespace detail

namespace gui {

#define GUI_WINDOW_TYPE(WINDOW, LABEL) class WINDOW;
GUI_WINDOW_TYPES
#undef GUI_WINDOW_TYPE

// Type info structs (mapping from window type to parameters)
template <typename T_Window>
struct WindowTypeInfo {};

#define GUI_WINDOW_TYPE(WINDOW, LABEL)                                         \
template <>                                                                    \
struct WindowTypeInfo<WINDOW> {                                                \
    static constexpr int    id      {detail::WindowTypeCounter::Id<WINDOW>()}; \
    static constexpr char   name[]  {#WINDOW};                                 \
    static constexpr char   label[] {LABEL};                                   \
};
GUI_WINDOW_TYPES
#undef GUI_WINDOW_TYPE

inline std::string windowTypeName(int windowTypeId)
{
    switch (windowTypeId) {
        #define GUI_WINDOW_TYPE(WINDOW, LABEL)   \
        case WindowTypeInfo<WINDOW>::id:         \
            return WindowTypeInfo<WINDOW>::name;
        GUI_WINDOW_TYPES
        #undef GUI_WINDOW_TYPE
        default: break;
    }
    throw std::runtime_error("Invalid window type id: " + std::to_string(windowTypeId) +
        " - have all window types been listed in WindowTypeUtils.hpp?");
}

// Callbacks for fetching window type using type name or ID,
// intended to be used with a generic (templated) lambda:
// windowTypeCallback(windowTypeName, []<typename T>(){ /* T is the window type here */ });
template <typename F>
void windowTypeNameCallback(const std::string& typeName, F&& f)
{
    #define GUI_WINDOW_TYPE(WINDOW, LABEL) \
    if (typeName == WindowTypeInfo<WINDOW>::name) \
        return f.template operator()<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
    throw std::runtime_error("No type name match for \"" + typeName + "\" found");
}

template <typename F>
void windowTypeIdCallback(int typeId, F&& f)
{
    #define GUI_WINDOW_TYPE(WINDOW, LABEL) \
    if (typeId == WindowTypeInfo<WINDOW>::id) \
        return f.template operator()<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
    throw std::runtime_error("No type name match for " + std::to_string(typeId) + " found");
}

// similar to the functions above, but this function calls the callback for each window type
template <typename F>
void windowForEachTypeCallback(F&& f)
{
    #define GUI_WINDOW_TYPE(WINDOW, LABEL) \
    f.template operator()<WINDOW>();
    GUI_WINDOW_TYPES
    #undef GUI_WINDOW_TYPE
}

} // namespace gui

#undef GUI_WINDOW_TYPES

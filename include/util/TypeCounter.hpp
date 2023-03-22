//
// Project: DooT2
// File: TypeCounter.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <type_traits>


namespace detail {

template <typename First, typename... Rest>
struct TypeCounter {
    template<typename T, typename U> struct IsSame : std::false_type {};
    template<typename T> struct IsSame<T, T> : std::true_type {};

    template <typename T>
    static consteval int Id() {
        if constexpr(IsSame<T, First>::value)
            return 0;
        else
            return TypeCounter<Rest...>::template Id<T>() + 1;
    }
};

} // namespace detail

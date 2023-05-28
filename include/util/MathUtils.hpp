//
// Project: DooT2
// File: MathUtils.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


namespace detail
{
    double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
    {
        return curr == prev
               ? curr
               : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }
} // namespace detail

/*
* Constexpr version of the square root
* Return value:
*   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
double constexpr constexprSqrt(double x)
{
    return x >= 0 && x < std::numeric_limits<double>::infinity()
           ? detail::sqrtNewtonRaphson(x, x, 0)
           : std::numeric_limits<double>::quiet_NaN();
}

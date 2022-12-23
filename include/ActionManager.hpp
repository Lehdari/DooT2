//
// Project: DooT2
// File: ActionManager.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include "ActionConverter.hpp"

#include <random>


class Heatmap;


class ActionManager {
public:
    ActionManager();

    // Add heatmap module
    void setHeatmap(const Heatmap* heatmap);

    gvizdoom::Action operator()();

private:
    ActionConverter<float>                  _actionConverter;

    // modules
    const Heatmap*                          _heatmap;

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;

    gvizdoom::Action generateRandomAction();
};

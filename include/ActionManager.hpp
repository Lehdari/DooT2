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
#include "MathTypes.hpp"

#include <random>


class Heatmap;


class ActionManager {
public:
    ActionManager();

    // Add heatmap module
    void setHeatmap(Heatmap* heatmap);

    void reset();

    gvizdoom::Action operator()(const Vec2f& playerPos);

    float                                   _heatmapDiff;

private:
    ActionConverter<float>                  _actionConverter;
    std::vector<float>                      _actionVector;

    // modules
    Heatmap*                                _heatmap;
    float                                   _heatmapSamplePrev;
    Vec2f                                   _pPrev;
    float                                   _sPrev;

    int                                     _forwardHoldTimer;
    int                                     _wallHitTimer;

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;

    void updateActionVector(float smoothing, float sigma);
};

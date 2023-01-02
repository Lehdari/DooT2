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
    struct UpdateParams {
        double  smoothing   {0.8f};
        double  sigma       {1.0f};
    };

    struct CallParams {
        Vec2f   playerPos   {0.0f, 0.0f};
    };

    ActionManager();

    // Add heatmap module
    void setHeatmap(Heatmap* heatmap);

    void reset();

    gvizdoom::Action operator()(const CallParams& callParams);

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


    void updateActionVector(const UpdateParams& params);
};

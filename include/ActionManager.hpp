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
#include <functional>


class HeatmapActionModule;


class ActionManager {
public:
    struct UpdateParams {
        double  smoothing   {0.8f};
        double  sigma       {1.0f};
        float   heatmapDiff {0.0f}; // TODO remove
    };

    struct CallParams {
        Vec2f   playerPos   {0.0f, 0.0f};
    };


    ActionManager();

    template <typename T_Module>
    void addModule(T_Module* module);

    void reset();

    gvizdoom::Action operator()(const CallParams& callParams);

private:
    using ModuleCall = std::function<void(const CallParams&, UpdateParams&)>;
    using ModuleReset = std::function<void()>;

    ActionConverter<float>                  _actionConverter;
    std::vector<float>                      _actionVector;
    std::vector<ModuleCall>                 _moduleCalls;
    std::vector<ModuleReset>                _moduleResets;

    Vec2f                                   _pPrev;
    float                                   _sPrev;

    int                                     _forwardHoldTimer;
    int                                     _wallHitTimer;

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;


    void updateActionVector(const UpdateParams& params);
};


template<typename T_Module>
void ActionManager::addModule(T_Module* module)
{
    _moduleCalls.push_back(std::bind(&T_Module::operator(), module,
        std::placeholders::_1, std::placeholders::_2));
    _moduleResets.push_back(std::bind(&T_Module::reset, module));
}

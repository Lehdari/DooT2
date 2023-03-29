//
// Project: DooT2
// File: ActionManager.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ActionManager.hpp"
#include "Constants.hpp"
#include "Heatmap.hpp"

using namespace doot2;
using namespace gvizdoom;


std::default_random_engine      ActionManager::_rnd       (1507715517);
std::normal_distribution<float> ActionManager::_rndNormal (0.0f, 1.0f);

int64_t ActionManager::moduleIdCounter  {0};


ActionManager::ActionManager() :
    _actionConverter    (),
    _actionVector       (actionVectorLength, 0.0f),
    _pPrev              (0.0f, 0.0f),
    _vPrev              (0.0f, 0.0f)
{
    // Setup action converter
    _actionConverter.setAngleIndex(0);
    _actionConverter.setKeyIndex(1, Action::Key::ACTION_FORWARD);
    _actionConverter.setKeyIndex(2, Action::Key::ACTION_BACK);
    _actionConverter.setKeyIndex(3, Action::Key::ACTION_LEFT);
    _actionConverter.setKeyIndex(4, Action::Key::ACTION_RIGHT);
    _actionConverter.setKeyIndex(5, Action::Key::ACTION_USE);
}

void ActionManager::reset()
{
    for (auto& moduleReset : _moduleResets)
        moduleReset();

    _actionVector = std::vector<float>(actionVectorLength, 0.0f);
    _updateParams = UpdateParams();
    _pPrev << 0.0f, 0.0f;
    _vPrev << 0.0f, 0.0f;
}

gvizdoom::Action ActionManager::operator()(const CallParams& callParams)
{
    // compute velocity and acceleration
    _updateParams.p = callParams.playerPos;
    _updateParams.v = _updateParams.p - _pPrev;
    _updateParams.a = _updateParams.v - _vPrev;
    _pPrev = _updateParams.p;
    _vPrev = _updateParams.v;

    // reset overwrite
    _updateParams.actionVectorOverwrite.clear();

    // process modules
    for (auto& moduleCall : _moduleCalls) {
        moduleCall(callParams, _updateParams, *this);
    }

    if (_updateParams.actionVectorOverwrite.empty())
        updateActionVector(_updateParams);
    else
        _actionVector = _updateParams.actionVectorOverwrite;

    return _actionConverter(_actionVector);
}

const std::vector<float>& ActionManager::getActionVector() const noexcept
{
    return _actionVector;
}

void ActionManager::updateActionVector(const UpdateParams& params)
{
    for (size_t i=0; i<actionVectorLength; ++i) {
        _actionVector[i] = params.smoothing*_actionVector[i] +
            (1.0f-params.smoothing)*params.sigma*_rndNormal(_rnd);
        _actionVector[i] = std::clamp(_actionVector[i], -1.0f, 1.0f);
    }

    // prevent forward / back and left / right being pressed at the same time
    if (_actionVector[1] > 0.0f && _actionVector[2] > 0.0f) {
        if (_actionVector[1] > _actionVector[2]) // bigger value dominates
            _actionVector[2] *= -1.0f;
        else
            _actionVector[1] *= -1.0f;
    }
    if (_actionVector[3] > 0.0f && _actionVector[4] > 0.0f) {
        if (_actionVector[3] > _actionVector[4]) // bigger value dominates
            _actionVector[4] *= -1.0f;
        else
            _actionVector[3] *= -1.0f;
    }
}

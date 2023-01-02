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
#include "Heatmap.hpp"


using namespace gvizdoom;


static constexpr size_t actionVectorLength = 6;


std::default_random_engine      ActionManager::_rnd       (1507715517);
std::normal_distribution<float> ActionManager::_rndNormal (0.0f, 1.0f);


ActionManager::ActionManager() :
    _actionConverter    (),
    _actionVector       (actionVectorLength, 0.0f),
    _heatmap            (nullptr),
    _forwardHoldTimer   (0),
    _wallHitTimer       (0),
    _pPrev              (0.0f, 0.0f),
    _sPrev              (0.0f)
{
    // Setup action converter
    _actionConverter.setAngleIndex(0);
    _actionConverter.setKeyIndex(1, Action::Key::ACTION_FORWARD);
    _actionConverter.setKeyIndex(2, Action::Key::ACTION_BACK);
    _actionConverter.setKeyIndex(3, Action::Key::ACTION_LEFT);
    _actionConverter.setKeyIndex(4, Action::Key::ACTION_RIGHT);
    _actionConverter.setKeyIndex(5, Action::Key::ACTION_USE);
}

void ActionManager::setHeatmap(Heatmap* heatmap)
{
    _heatmap = heatmap;
}

void ActionManager::reset()
{
    if (_heatmap)
        _heatmap->reset();

    _actionVector = std::vector<float>(actionVectorLength, 0.0f);
    _forwardHoldTimer = 0;
    _wallHitTimer = 0;
    _pPrev << 0.0f, 0.0f;
    _sPrev = 0.0f;
}

gvizdoom::Action ActionManager::operator()(const CallParams& callParams)
{
    UpdateParams updateParams;

    if (_heatmap) {
        float heatmapSample = _heatmap->sample(callParams.playerPos, true);
        _heatmapDiff = heatmapSample-_heatmapSamplePrev;
        _heatmapSamplePrev = heatmapSample;
    }

    // velocity, speed and acceleration
    Vec2f v = callParams.playerPos - _pPrev;
    float s = v.norm();
    float a = s-_sPrev;
    _pPrev = callParams.playerPos;
    _sPrev = s;

    // count consequent forward presses
    if (_actionVector[1] > 0.0f && _actionVector[2] <= 0.0f)
        _forwardHoldTimer++;
    else
        _forwardHoldTimer = 0;

    if (_wallHitTimer == 0) {
        // detect when forward motion has hit a wall
        if (_forwardHoldTimer>15 && a<-0.5f) {
            _wallHitTimer = 60;
        }
        else if (_heatmapDiff > 0.0f) { // randomize action in case we're moving towards areas
            // which has been visited more (reuse actions otherwise)
            if (_heatmapDiff > 0.15f) {
                // invert action in case heatmap value grows rapidly (we're approaching region
                // that has been visited often)
                for (int i=0; i<actionVectorLength; ++i)
                    _actionVector[i] *= -1.0f;
            }
            updateActionVector(updateParams);
        }
    }
    else {
        // in case wall has been hit, cycle use to open potential door
        // and keep moving forward for 60 tics
        if (_wallHitTimer == 60)
            _actionVector = {0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
        else if (_wallHitTimer == 59)
            _actionVector = {0.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

        --_wallHitTimer;
    }

    return _actionConverter(_actionVector);
}

void ActionManager::updateActionVector(const UpdateParams& params)
{
    for (size_t i=0; i<actionVectorLength; ++i) {
        _actionVector[i] = params.smoothing*_actionVector[i] +
            (1.0f-params.smoothing)*params.sigma*_rndNormal(_rnd);
        _actionVector[i] = std::clamp(_actionVector[i], -1.0f, 1.0f);
    }
}

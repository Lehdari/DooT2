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


using namespace gvizdoom;


std::default_random_engine      ActionManager::_rnd       (1507715517);
std::normal_distribution<float> ActionManager::_rndNormal (0.0f, 0.666f);


ActionManager::ActionManager() :
    _actionConverter()
{
    // Setup action converter
    _actionConverter.setAngleIndex(0);
    _actionConverter.setKeyIndex(1, Action::Key::ACTION_FORWARD);
    _actionConverter.setKeyIndex(2, Action::Key::ACTION_BACK);
    _actionConverter.setKeyIndex(3, Action::Key::ACTION_LEFT);
    _actionConverter.setKeyIndex(4, Action::Key::ACTION_RIGHT);
    _actionConverter.setKeyIndex(5, Action::Key::ACTION_USE);
}

gvizdoom::Action ActionManager::operator()() {
    return generateRandomAction();
}

gvizdoom::Action ActionManager::generateRandomAction()
{
    constexpr size_t actionVectorLength = 6;
    constexpr float smoothing = 0.75f; // smoothing effectively causes discrete actions to "stick"
    static std::vector<float> actionVector(actionVectorLength, 0.0f);

    for (size_t i=0; i<actionVectorLength; ++i) {
        actionVector[i] = smoothing*actionVector[i] + (1.0f-smoothing)*_rndNormal(_rnd);
    }

    return _actionConverter(actionVector);
}

//
// Project: DooT2
// File: DoorTraversalActionModule.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ActionManager.hpp"


class DoorTraversalActionModule {
public:
    struct State {
        int forwardHoldTimer    {0};
        int wallHitTimer        {0};
    };

    DoorTraversalActionModule();

    void reset();

    void operator()(
        const ActionManager::CallParams& callParams,
        ActionManager::UpdateParams& updateParams,
        ActionManager& actionManager);

    State   state;
};
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


// ActionManager module for simple door traversal procedures (including escape
// from the oblige start rooms)
// It also provides a minimal ActionManager module implementation example with
// only the absolutely necessary components
class DoorTraversalActionModule {
public:
    // Public state struct which is fetchable via ActionManager::getModuleState
    struct State {
        int forwardHoldTimer    {0};
        int wallHitTimer        {0};
    };

    DoorTraversalActionModule(bool useWallHitDoorTraversal = true);

    // Reset is called in game restarts. Should reset the state
    void reset();

    // Iteration. Called by ActionManager during its call
    // callParams:      Call parameters provided into the ActionManager call. Include
    //                  information about the game state
    // updateParams:    Parameters used for producing the next action. Primary product
    //                  of the module calling process
    // actionManager:   Reference to the ActionManager instance performing the call
    void operator()(
        const ActionManager::CallParams& callParams,
        ActionManager::UpdateParams& updateParams,
        ActionManager& actionManager);

    State   state; // public state member

private:
    bool    _useWallHitDoorTraversal;
};
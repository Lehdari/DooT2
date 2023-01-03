//
// Project: DooT2
// File: DoorTraversalActionModule.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "DoorTraversalActionModule.hpp"
#include "HeatmapActionModule.hpp"


static std::vector<std::vector<float>> startRoomEscapeSequence = [](){
    std::vector<std::vector<float>> v;
    for (int i=0; i<5; ++i)
        v.push_back({0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f}); // move forward
    v.push_back({0.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f}); // press use
    for (int i=0; i<70; ++i)
        v.push_back({0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f}); // move forward
    return v;
}();


DoorTraversalActionModule::DoorTraversalActionModule()
{
}

void DoorTraversalActionModule::reset()
{
    state = State();
}

void DoorTraversalActionModule::operator()(
    const ActionManager::CallParams& callParams,
    ActionManager::UpdateParams& updateParams,
    ActionManager& actionManager
) {
    auto& actionVector = actionManager.getActionVector();
    auto& heatmapState = actionManager.getModuleState<HeatmapActionModule>();
    assert(updateParams.actionVectorOverwrite.empty());

    // escape from the start room
    if (callParams.frameId < startRoomEscapeSequence.size()) {
        updateParams.actionVectorOverwrite = startRoomEscapeSequence[callParams.frameId];
        return;
    }

    // count consequent forward presses
    if (actionVector[1] > 0.0f)
        state.forwardHoldTimer++;
    else
        state.forwardHoldTimer = 0;

    float dot = updateParams.v.dot(updateParams.a);
    if (state.wallHitTimer <= 0) {
        // detect when forward motion has hit a wall
        if (state.forwardHoldTimer>10 && dot < -5.0f) {
            state.wallHitTimer = 60;
        }
        else if (heatmapState.diff > 0.15f) {
            // invert action in case heatmap value grows rapidly (we're approaching region
            // that has been visited often)
            updateParams.actionVectorOverwrite.resize(actionVector.size());
            for (int i=0; i<actionVector.size(); ++i)
                updateParams.actionVectorOverwrite[i] = -1.0f*actionVector[i];
        }
    }
    else {
        // in case wall has been hit, cycle use to open potential door
        // and keep moving forward for 60 tics
        if (state.wallHitTimer == 60)
            updateParams.actionVectorOverwrite = {0.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f}; // TODO generalize
        else
            updateParams.actionVectorOverwrite = {0.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f};

        --state.wallHitTimer;
    }
}

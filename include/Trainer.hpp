//
// Project: DooT2
// File: Trainer.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include "ActionManager.hpp"
#include "HeatmapActionModule.hpp"
#include "DoorTraversalActionModule.hpp"
#include "SequenceStorage.hpp"

#include <opencv2/core/mat.hpp>
#include "gvizdoom/Action.hpp"

#include <atomic>
#include <random>


class Model;


class Trainer {
public:
    Trainer(Model* model);
    // TODO RO5
    ~Trainer();

    void loop();
    void quit();
private:
    using Rnd = std::default_random_engine;
    Rnd                         _rnd;
    std::atomic_bool            _quit;
    ActionManager               _actionManager;
    HeatmapActionModule         _heatmapActionModule;
    DoorTraversalActionModule   _doorTraversalActionModule;
    SequenceStorage             _sequenceStorage;
    cv::Mat                     _positionPlot;
    Vec2f                       _initPlayerPos;

    size_t                      _frameId;
    size_t                      _batchEntryId;
    bool                        _newPatchReady;

    Model*                      _model;

    void nextMap(); // proceed to next map
};

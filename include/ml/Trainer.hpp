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
#include "DoorTraversalActionModule.hpp"
#include "HeatmapActionModule.hpp"
#include "util/SequenceStorage.hpp"
#include "util/SingleBuffer.hpp"

#include "gvizdoom/Action.hpp"

#include <atomic>
#include <random>



namespace ml {

class Model;


class Trainer {
public:
    Trainer(Model* model, uint32_t batchSizeIn, size_t sequenceLengthIn);
    ~Trainer();
    Trainer(const Trainer&) = delete;
    Trainer(Trainer&&) noexcept = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    void loop();
    void quit();

    const SingleBuffer<Image<uint8_t>>::ReadHandle getFrameReadHandle();
private:
    using Rnd = std::default_random_engine;
    Rnd                             _rnd;
    std::atomic_bool                _quit;
    ActionManager                   _actionManager;
    HeatmapActionModule             _heatmapActionModule;
    DoorTraversalActionModule       _doorTraversalActionModule;
    SequenceStorage                 _sequenceStorage;
    cv::Mat                         _positionPlot;
    Vec2f                           _initPlayerPos;
    SingleBuffer<Image<uint8_t>>    _frame;

    size_t                          _frameId;
    size_t                          _batchEntryId;
    bool                            _newPatchReady;

    Model*                          _model;

    void nextMap(); // proceed to next map
};

} // namespace ml
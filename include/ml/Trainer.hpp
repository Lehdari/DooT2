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


#include "ActionConverter.hpp"
#include "util/Image.hpp"
#include "util/SequenceStorage.hpp"
#include "util/SingleBuffer.hpp"
#include "Constants.hpp"

#include "gvizdoom/Action.hpp"

#include <atomic>
#include <random>



namespace ml {

class Model;


class Trainer {
public:
    // model:           model to be trained
    // agentModel:      model to be used for the agent
    // encoderModel:    model used for encoding the frames (if nullptr, raw frames will be stored
    //                  into the sequence storage instead)
    Trainer(
        Model* model,
        Model* agentModel,
        Model* encoderModel = nullptr,
        uint32_t batchSizeIn = doot2::batchSize,
        size_t sequenceLengthIn = doot2::sequenceLength);
    // Trainer(Model* modelEnc, Model* modelAc, uint32_t batchSizeIn, size_t sequenceLengthIn);
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
    ActionConverter<float>          _actionConverter;
    SequenceStorage                 _sequenceStorage;
    SingleBuffer<Image<uint8_t>>    _frame;

    size_t                          _frameId;
    size_t                          _batchEntryId;
    bool                            _newPatchReady;

    Model*                          _model;
    Model*                          _agentModel;
    Model*                          _encoderModel;

    // Model*                          _modelEnc;
    // Model*                          _modelAc;

    void nextMap(); // proceed to next map
};

} // namespace ml

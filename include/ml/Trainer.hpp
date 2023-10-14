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
#include "ml/TrainingInfo.hpp"
#include "util/DoubleBuffer.hpp"
#include "util/Image.hpp"
#include "util/SequenceCache.hpp"
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
        Model* agentModel,
        Model* encoderModel = nullptr,
        uint32_t batchSizeIn = doot2::batchSize,
        size_t sequenceLengthIn = doot2::sequenceLength);
    ~Trainer();
    Trainer(const Trainer&) = delete;
    Trainer(Trainer&&) noexcept = delete;
    Trainer& operator=(const Trainer&) = delete;
    Trainer& operator=(Trainer&&) = delete;

    void loop();
    void quit();
    bool isFinished() const noexcept;
    void pause();
    void unpause();
    bool isPaused() const noexcept;

    void configureExperiment(nlohmann::json&& experimentConfig);
    void setupExperiment(); // called before starting the training loop
    void saveExperiment();

    // Access the model that is being trained
    Model* getModel();
    TrainingInfo* getTrainingInfo();
    nlohmann::json* getExperimentConfig();

    const SingleBuffer<Image<uint8_t>>::ReadHandle getFrameReadHandle();
private:
    using Rnd = std::default_random_engine;
    Rnd                             _rnd;
    std::atomic_bool                _quit;
    std::atomic_bool                _paused;
    std::mutex                      _pauseMutex;
    std::condition_variable         _pauseCV;
    std::atomic_bool                _finished; // loop() finished, waiting to join
    ActionConverter<float>          _actionConverter;
    DoubleBuffer<SequenceStorage>   _sequenceStorage;
    SequenceCache                   _sequenceCache;
    SingleBuffer<Image<uint8_t>>    _frameRGB;
    std::vector<float>              _frameYUVData;
    Image<float>                    _frameYUV;
    nlohmann::json                  _experimentConfig;
    TrainingInfo                    _trainingInfo;

    std::unique_ptr<Model>          _model;
    Model*                          _agentModel;
    Model*                          _encoderModel;

    std::set<int>                   _visitedMaps; // for keeping track of maps visited in current batch to avoid duplicates
    Vec3f                           _playerInitPos;
    float                           _playerDistanceThreshold; // distance from start to start recording

    void refreshSequenceStorage(int epoch, bool evaluation = false); // load new sequences to _sequenceStorage
    void recordTrainingSequences();
    void recordEvaluationSequences();
    bool isEvaluationEpoch(int epoch) const;
    bool terminationEpochReached(int epoch) const;
    void updateCache(int epoch);

    bool gameStep();
    bool startRecording();
    int nextMap(bool evaluation = false); // proceed to next map, returns the map ID
    void evaluateModel();
    void createExperimentDirectories() const;
    void loadBaseExperimentTrainingInfo();
};

} // namespace ml

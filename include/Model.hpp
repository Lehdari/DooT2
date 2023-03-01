//
// Project: DooT2
// File: Model.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "nlohmann/json.hpp"

#include "Types.hpp"
#include "SequenceStorage.hpp"
#include "DoubleBuffer.hpp"
#include "SingleBuffer.hpp"
#include "TimeSeries.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>


using TrainingState = nlohmann::json;


// Interface class for models
// It provides facilities for synchronous and asynchronous (threaded) training
// Implement pure virtual functions trainImpl and infer in the derived class
class Model {
public:
    Model();

    // Train the model
    void train(SequenceStorage& storage);

    // Train the model in separate thread, makes hard copy of the storage, returns immediately
    void trainAsync(SequenceStorage storage);

    // Check if the asynchronous training is finished
    bool trainingFinished() const noexcept;

    // Wait for the asynchronous training to finish
    void waitForTrainingFinished() noexcept;

    virtual void infer(const TensorVector& input, TensorVector& output) = 0;

    void abortTraining() noexcept;

    // Information about training
    DoubleBuffer<TrainingState> trainingState;
    SingleBuffer<TimeSeries>    timeSeries;

private:
    std::mutex              _trainingMutex;
    std::condition_variable _trainingCv;
    bool                    _trainingFinished;

protected:
    std::atomic_bool        _abortTraining{false};

    // Required for storing the copy of the storage made in trainAsync
    void trainAsyncThreadWrapper(SequenceStorage&& storage);

    virtual void trainImpl(SequenceStorage& storage) = 0;
};

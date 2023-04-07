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

#include "util/Types.hpp"
#include "util/Image.hpp"
#include "util/SequenceStorage.hpp"
#include "util/DoubleBuffer.hpp"
#include "util/SingleBuffer.hpp"
#include "util/TimeSeries.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>


namespace ml {

struct TrainingInfo;


// Interface class for models, which provides facilities for synchronous and
// asynchronous (threaded) training.
// Implement pure virtual functions reset, infer and trainImpl in the derived class.
// Optionally, the default functionality of setTrainingInfo can be overridden.
class Model {
public:
    Model();
    virtual ~Model() = default;

    // Set pointer to training info
    virtual void setTrainingInfo(TrainingInfo* trainingInfo);

    // Reset the model - will be called in start of each sequence
    virtual void reset() = 0;

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

private:
    std::mutex              _trainingMutex;
    std::condition_variable _trainingCv;
    bool                    _trainingFinished;

protected:
    TrainingInfo*           _trainingInfo;

    std::atomic_bool        _abortTraining{false};

    // Required for storing the copy of the storage made in trainAsync
    void trainAsyncThreadWrapper(SequenceStorage&& storage);

    virtual void trainImpl(SequenceStorage& storage) = 0;
};

} // namespace ml

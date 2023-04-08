//
// Project: DooT2
// File: Model.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/Model.hpp"
#include "ml/TrainingInfo.hpp"


using namespace ml;


Model::Model(nlohmann::json* experimentConfig) :
    _trainingFinished   (true),
    _experimentConfig   (experimentConfig),
    _trainingInfo       (nullptr)
{
    if (_experimentConfig != nullptr) {
        assert(_experimentConfig->contains("experiment_root"));
        assert(_experimentConfig->contains("model_config"));
    }
}

void Model::setTrainingInfo(TrainingInfo* trainingInfo)
{
    _trainingInfo = trainingInfo;
}

void Model::reset()
{
}

void Model::save()
{
}

void Model::train(SequenceStorage& storage)
{
    std::unique_lock<std::mutex> lock(_trainingMutex);
    _trainingCv.wait(lock, [&]{ return _trainingFinished; });
    _trainingFinished = false;
    _abortTraining = false;

    trainImpl(storage);

    _trainingFinished = true;
    lock.unlock();
    _trainingCv.notify_all();
}

void Model::trainAsync(SequenceStorage storage)
{
    std::thread t(&Model::trainAsyncThreadWrapper, this, std::move(storage));
    t.detach();
}

bool Model::trainingFinished() const noexcept
{
    return _trainingFinished;
}

void Model::waitForTrainingFinished() noexcept
{
    std::unique_lock<std::mutex> lock(_trainingMutex);
    _trainingCv.wait(lock, [&]{ return _trainingFinished; });
}

void Model::trainAsyncThreadWrapper(SequenceStorage&& storage)
{
    train(storage);
}

void Model::abortTraining() noexcept
{
    _abortTraining = true;
}
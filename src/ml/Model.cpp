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


Model::Model() :
    _trainingFinished   (true),
    _trainingInfo       (nullptr),
    _nAsyncCalls        (0)
{
}

void Model::setTrainingInfo(TrainingInfo* trainingInfo)
{
    _trainingInfo = trainingInfo;
}

void Model::reset()
{
}

void Model::save(const std::filesystem::path& subdir)
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

void Model::trainAsync(const DoubleBuffer<SequenceStorage>::ReadHandle&& storageReadHandle)
{
    ++_nAsyncCalls;
    // const_cast is required because pytorch does not allow const buffers to be mapped into tensors
    std::thread t(&Model::trainAsyncThreadWrapper, this,
        std::ref(const_cast<SequenceStorage&>(*storageReadHandle)));
    t.detach();
}

bool Model::trainingFinished() const noexcept
{
    return _trainingFinished && _nAsyncCalls == 0;
}

void Model::waitForTrainingFinished() noexcept
{
    std::unique_lock<std::mutex> lock(_trainingMutex);
    _trainingCv.wait(lock, [&]{ return trainingFinished(); });
}

void Model::trainAsyncThreadWrapper(SequenceStorage& storage)
{
    train(storage);
    if (--_nAsyncCalls < 0)
        throw std::runtime_error("Model::_nAsyncCalls < 0: BUG"); // something's gone terribly wrong with the async threading
}

void Model::abortTraining() noexcept
{
    _abortTraining = true;
}
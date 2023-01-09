//
// Project: DooT2
// File: SequenceStorage.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "SequenceStorage.hpp"

#include <cassert>


SequenceStorage::BatchHandle::BatchHandle(
    gvizdoom::Action* actions,
    Image<float>* frames,
    double* rewards
) :
    actions (actions),
    frames  (frames),
    rewards (rewards)
{
}

SequenceStorage::ConstBatchHandle::ConstBatchHandle(
    const gvizdoom::Action* actions,
    const Image<float>* frames,
    const double* rewards
) :
    actions (actions),
    frames  (frames),
    rewards (rewards)
{
}

SequenceStorage::SequenceStorage(const Settings& settings) :
    _settings   (settings),
    _actions    (_settings.length*_settings.batchSize),
    _frameData  (_settings.length*_settings.batchSize*_settings.frameWidth*_settings.frameHeight
                 *getImageFormatNChannels(_settings.frameFormat)),
    _rewards    (_settings.length*_settings.batchSize)
{
    std::size_t size = _settings.length*_settings.batchSize;
    initializeFrames(size);
}

SequenceStorage::SequenceStorage(const SequenceStorage& other) :
    _settings   (other._settings),
    _actions    (other._actions),
    _frameData  (other._frameData),
    _rewards    (other._rewards)
{
    initializeFrames(other._frames.size());
}

SequenceStorage::SequenceStorage(SequenceStorage&& other) noexcept :
    _settings   (std::move(other._settings)),
    _actions    (std::move(other._actions)),
    _frameData  (std::move(other._frameData)),
    _rewards    (std::move(other._rewards))
{
    initializeFrames(other._frames.size());
}

SequenceStorage& SequenceStorage::operator=(const SequenceStorage& other)
{
    _settings = other._settings;
    _actions = other._actions;
    _frameData = other._frameData;
    _rewards = other._rewards;

    initializeFrames(other._frames.size());

    return *this;
}

SequenceStorage& SequenceStorage::operator=(SequenceStorage&& other) noexcept
{
    _settings = std::move(other._settings);
    _actions = std::move(other._actions);
    _frameData = std::move(other._frameData);
    _rewards = std::move(other._rewards);

    initializeFrames(other._frames.size());

    return *this;
}

SequenceStorage::BatchHandle SequenceStorage::operator[](std::size_t id)
{
    assert(id < _settings.length);

    return {
        &_actions[id*_settings.batchSize],
        &_frames[id*_settings.batchSize],
        &_rewards[id*_settings.batchSize]
    };
}

SequenceStorage::ConstBatchHandle SequenceStorage::operator[](std::size_t id) const noexcept
{
    assert(id < _settings.length);

    return {
        &_actions[id*_settings.batchSize],
        &_frames[id*_settings.batchSize],
        &_rewards[id*_settings.batchSize]
    };
}

const SequenceStorage::Settings& SequenceStorage::settings() const noexcept
{
    return _settings;
}

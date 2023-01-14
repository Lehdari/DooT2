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
    float** encodings,
    double* rewards,
    float* frameData,
    float* encodingData,
    const SequenceStorage::Settings& settings
) :
    actions         (actions),
    frames          (frames),
    encodings       (encodings),
    rewards         (rewards),
    _frameData      (frameData),
    _encodingData   (encodingData),
    _settings       (settings)
{
}

const torch::Tensor SequenceStorage::BatchHandle::mapPixelData()
{
    if (_settings.hasFrames) {
        return torch::from_blob(
            _frameData,
            { _settings.batchSize, _settings.frameHeight, _settings.frameWidth,
            getImageFormatNChannels(_settings.frameFormat) },
            torch::TensorOptions().device(torch::kCPU)
        );
    }
    else {
        return torch::empty({}, torch::TensorOptions().device(torch::kCPU));
    }
}

const torch::Tensor SequenceStorage::BatchHandle::mapEncodingData()
{
    if (_settings.hasEncodings) {
        return torch::from_blob(
            _encodingData,
            { _settings.batchSize, _settings.encodingLength },
            torch::TensorOptions().device(torch::kCPU)
        );
    }
    else {
        return torch::empty({}, torch::TensorOptions().device(torch::kCPU));
    }
}

SequenceStorage::ConstBatchHandle::ConstBatchHandle(
    const gvizdoom::Action* actions,
    const Image<float>* frames,
    const float* const* encodings,
    const double* rewards
) :
    actions     (actions),
    frames      (frames),
    encodings   (encodings),
    rewards     (rewards)
{
}

SequenceStorage::SequenceStorage(const Settings& settings) :
    _settings       (settings),
    _frameSize      (_settings.frameWidth*_settings.frameHeight*getImageFormatNChannels(_settings.frameFormat)),
    _actions        (_settings.length*_settings.batchSize),
    _frameData      (_settings.hasFrames ? _settings.length*_settings.batchSize*_frameSize : 0),
    _encodings      (_settings.hasEncodings ? _settings.length*_settings.batchSize : 0),
    _encodingData   (_settings.hasEncodings ? _settings.length*_settings.batchSize*_settings.encodingLength : 0),
    _rewards        (_settings.length*_settings.batchSize)
{
    std::size_t size = _settings.length*_settings.batchSize;
    if (_settings.hasFrames)
        initializeFrames(size);
    if (_settings.hasEncodings)
        initializeEncodings(size);
}

SequenceStorage::SequenceStorage(const SequenceStorage& other) :
    _settings       (other._settings),
    _frameSize      (other._frameSize),
    _actions        (other._actions),
    _frameData      (other._frameData),
    _encodingData   (other._encodingData),
    _rewards        (other._rewards)
{
    if (_settings.hasFrames)
        initializeFrames(other._frames.size());
    if (_settings.hasEncodings)
        initializeEncodings(other._encodings.size());
}

SequenceStorage::SequenceStorage(SequenceStorage&& other) noexcept :
    _settings       (std::move(other._settings)),
    _frameSize      (std::move(other._frameSize)),
    _actions        (std::move(other._actions)),
    _frameData      (std::move(other._frameData)),
    _encodingData   (std::move(other._encodingData)),
    _rewards        (std::move(other._rewards))
{
    if (_settings.hasFrames)
        initializeFrames(other._frames.size());
    if (_settings.hasEncodings)
        initializeEncodings(other._encodings.size());
}

SequenceStorage& SequenceStorage::operator=(const SequenceStorage& other)
{
    _settings = other._settings;
    _frameSize = other._frameSize;
    _actions = other._actions;
    _frameData = other._frameData;
    _encodingData = other._encodingData;
    _rewards = other._rewards;

    if (_settings.hasFrames)
        initializeFrames(other._frames.size());
    if (_settings.hasEncodings)
        initializeEncodings(other._encodings.size());

    return *this;
}

SequenceStorage& SequenceStorage::operator=(SequenceStorage&& other) noexcept
{
    _settings = std::move(other._settings);
    _frameSize = std::move(other._frameSize);
    _actions = std::move(other._actions);
    _frameData = std::move(other._frameData);
    _encodingData = std::move(other._encodingData);
    _rewards = std::move(other._rewards);

    if (_settings.hasFrames)
        initializeFrames(other._frames.size());
    if (_settings.hasEncodings)
        initializeEncodings(other._encodings.size());

    return *this;
}

SequenceStorage::BatchHandle SequenceStorage::operator[](std::size_t id)
{
    assert(id < _settings.length);

    return {
        &_actions[id*_settings.batchSize],
        _settings.hasFrames ? &_frames[id*_settings.batchSize] : nullptr,
        _settings.hasEncodings ? &_encodings[id*_settings.batchSize] : nullptr,
        &_rewards[id*_settings.batchSize],
        _settings.hasFrames ? &_frameData[id*_settings.batchSize*_frameSize] : nullptr,
        _settings.hasEncodings ? &_encodingData[id*_settings.batchSize*_settings.encodingLength] : nullptr,
        _settings
    };
}

SequenceStorage::ConstBatchHandle SequenceStorage::operator[](std::size_t id) const noexcept
{
    assert(id < _settings.length);

    return {
        &_actions[id*_settings.batchSize],
        _settings.hasFrames ? &_frames[id*_settings.batchSize] : nullptr,
        _settings.hasEncodings ? &_encodings[id * _settings.batchSize] : nullptr,
        &_rewards[id*_settings.batchSize]
    };
}

const torch::Tensor SequenceStorage::mapPixelData()
{
    if (_settings.hasFrames) {
        return torch::from_blob(
            _frameData.data(),
            { (long)_settings.length, _settings.batchSize,
                _settings.frameHeight, _settings.frameWidth,
                getImageFormatNChannels(_settings.frameFormat) },
            torch::TensorOptions().device(torch::kCPU)
        );
    }
    else {
        return torch::empty({}, torch::TensorOptions().device(torch::kCPU));
    }
}

const torch::Tensor SequenceStorage::mapEncodingData()
{
    if (_settings.hasEncodings) {
        return torch::from_blob(
            _encodingData.data(),
            { (long)_settings.length, _settings.batchSize, _settings.encodingLength },
            torch::TensorOptions().device(torch::kCPU)
        );
    }
    else {
        return torch::empty({}, torch::TensorOptions().device(torch::kCPU));
    }
}

const SequenceStorage::Settings& SequenceStorage::settings() const noexcept
{
    return _settings;
}

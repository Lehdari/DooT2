//
// Project: DooT2
// File: SequenceStorage.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Image.hpp"

#include <gvizdoom/Action.hpp>

class SequenceStorage {
public:
    struct Settings {
        uint32_t    batchSize;
        std::size_t length;     // max sequence length
        uint32_t    frameWidth;
        uint32_t    frameHeight;
        ImageFormat frameFormat;
    };

    class BatchHandle {
    public:
        gvizdoom::Action* const actions;
        Image<float>* const     frames;
        double* const           rewards;

        friend class SequenceStorage;

    private:
        BatchHandle(
            gvizdoom::Action* actions,
            Image<float>* frames,
            double* rewards);
    };

    class ConstBatchHandle {
    public:
        const gvizdoom::Action* const   actions;
        const Image<float>* const       frames;
        const double* const             rewards;

        friend class SequenceStorage;

    private:
        ConstBatchHandle(
            const gvizdoom::Action* actions,
            const Image<float>* frames,
            const double* rewards);
    };


    SequenceStorage(const Settings& settings);
    SequenceStorage(const SequenceStorage& other);
    SequenceStorage(SequenceStorage&& other) noexcept;
    SequenceStorage& operator=(const SequenceStorage& other);
    SequenceStorage& operator=(SequenceStorage&& other) noexcept;

    // Access a batch
    BatchHandle operator[](std::size_t id);
    ConstBatchHandle operator[](std::size_t id) const noexcept;

    const Settings& settings() const noexcept;

private:
    Settings                        _settings;

    // Sequence data vectors
    std::vector<gvizdoom::Action>   _actions;
    std::vector<Image<float>>       _frames;
    std::vector<float>              _frameData; // pixel data storage for _frames
    std::vector<double>             _rewards;

    inline void initializeFrames(std::size_t size);
};


void SequenceStorage::initializeFrames(std::size_t size)
{
    std::size_t frameSize = _settings.frameWidth*_settings.frameHeight
        *getImageFormatNChannels(_settings.frameFormat);

    _frames.reserve(size);
    for (std::size_t i=0; i<size; ++i) {
        // Initialize images with pixel data stored in _frameData
        _frames.emplace_back(_settings.frameWidth, _settings.frameHeight, _settings.frameFormat,
            &_frameData[i*frameSize]);
    }
}

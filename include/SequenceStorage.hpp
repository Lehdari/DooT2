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
#include "TensorUtils.hpp"

#include <gvizdoom/Action.hpp>


class SequenceStorage {
public:
    struct Settings {
        uint32_t    batchSize;
        std::size_t length;     // max sequence length

        bool        hasFrames       {true};     // Does the storage contain frames?
        bool        hasEncodings    {false};    // Does the storage contain frame encodings?

        uint32_t    frameWidth;
        uint32_t    frameHeight;
        ImageFormat frameFormat;
        uint32_t    encodingLength;
    };

    class BatchHandle {
    public:
        gvizdoom::Action* const actions;
        Image<float>* const     frames;
        float* const* const     encodings; // const ptr to const ptr to mutable float
        double* const           rewards;

        // Map pixel data to a torch tensor (BHWC)
        const torch::Tensor mapPixelData();

        // Map encoding data to a torch tensor (BW)
        const torch::Tensor mapEncodingData();

        // Map rewards to a torch tensor (B)
        const torch::Tensor mapRewards();

        friend class SequenceStorage;

    private:
        BatchHandle(
            gvizdoom::Action* actions,
            Image<float>* frames,
            float** encodings,
            double* rewards,
            float* frameData,
            float* encodingData,
            const SequenceStorage::Settings& settings);

        float* const                        _frameData;
        float* const                        _encodingData;

        const SequenceStorage::Settings&    _settings;
    };

    class ConstBatchHandle {
    public:
        const gvizdoom::Action* const   actions;
        const Image<float>* const       frames;
        const float* const* const       encodings; // const ptr to const ptr to const float
        const double* const             rewards;

        friend class SequenceStorage;

    private:
        ConstBatchHandle(
            const gvizdoom::Action* actions,
            const Image<float>* frames,
            const float* const* encodings,
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

    // Map pixel data to a torch tensor (LBHWC)
    const torch::Tensor mapPixelData();

    // Map encoding data to a torch tensor (LBW)
    const torch::Tensor mapEncodingData();

    // Map rewards to a torch tensor (LB)
    const torch::Tensor mapRewards();

    const Settings& settings() const noexcept;

    // Reinitialize all data to default values (0 and such)
    void reset();

private:
    Settings                        _settings;
    uint64_t                        _frameSize;     // size of a frame in elements

    // Sequence data vectors
    std::vector<gvizdoom::Action>   _actions;
    std::vector<Image<float>>       _frames;
    std::vector<float>              _frameData;     // pixel data storage for _frames
    std::vector<float*>             _encodings;     // pointers to starting points of each encoding in the _encodingData vector
    std::vector<float>              _encodingData;  // frame encodings in one long vector, similar to _frameData
    std::vector<double>             _rewards;

    inline void initializeFrames(std::size_t size);
    inline void initializeEncodings(std::size_t size);
};


void SequenceStorage::initializeFrames(std::size_t size)
{
    _frames.clear();
    _frames.reserve(size);
    for (std::size_t i=0; i<size; ++i) {
        // Initialize images with pixel data stored in _frameData
        _frames.emplace_back(_settings.frameWidth, _settings.frameHeight, _settings.frameFormat,
            &_frameData[i*_frameSize]);
    }
}

void SequenceStorage::initializeEncodings(std::size_t size)
{
    _encodings.resize(size);
    for (std::size_t i=0; i<size; ++i) {
        // point to start of each encoding in _encodingData
        _encodings[i] = &_encodingData[i*_settings.encodingLength];
    }
}

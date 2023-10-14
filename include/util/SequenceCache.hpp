//
// Project: DooT2
// File: SequenceCache.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>
#include <unordered_map>

#include "Image.hpp"


class SequenceStorage;
template <typename T_Data>
class Image;


class SequenceCache {
public:
    enum class Type {
        FRAME_ENCODING_TRAINING_NORMAL,
        FRAME_ENCODING_TRAINING_HARD, // TODO training cache formed from entries the model has performed poorly on
        FRAME_ENCODING_EVALUATION
    };

    explicit SequenceCache(int maxLoadingThreads = 16);

    void setPath(const std::filesystem::path& cachePath);

    uint64_t nAvailableSequences(Type cacheType) const;

    void recordEntry(
        Type cacheType,
        std::string_view sequenceName,
        int batchId,
        int frameId,
        const Image<uint8_t>& image);
    //void recordEntry(Type cacheType, int batchId, int frameId, ...); // TODO add these for other entry types
    void finishRecord(Type cacheType);
    void deleteOldest(Type cacheType);

    void loadFramesToStorage(
        SequenceStorage& sequenceStorage,
        Type cacheType,
        std::string_view sequenceName,
        int nSequences,
        int offset);

private:
    static std::unordered_map<Type, std::filesystem::path>  typeSubPath;

    int                                     _nLoadingThreads;
    std::filesystem::path                   _cachePath;
    std::unordered_map<Type, std::string>   _recordSequenceNames;
    std::vector<std::vector<float>>         _frameYUVData;
    std::vector<Image<float>>               _frameYUV;
    std::vector<int64_t>                    _frameShape;

    void newRecordSequenceName(Type cacheType);
    void initializeYUVFrame(int width, int height);
};

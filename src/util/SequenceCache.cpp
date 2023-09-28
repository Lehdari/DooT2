//
// Project: DooT2
// File: SequenceCache.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "util/SequenceCache.hpp"
#include "util/SequenceStorage.hpp"
#include "util/Image.hpp"
#include "Constants.hpp"


namespace fs = std::filesystem;


std::unordered_map<SequenceCache::Type, std::filesystem::path> SequenceCache::typeSubPath = [](){
    std::unordered_map<SequenceCache::Type, std::filesystem::path> map;
    map[SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL] = fs::path("frame_encoding") / "training_normal";
    map[SequenceCache::Type::FRAME_ENCODING_TRAINING_HARD] = fs::path("frame_encoding") / "training_hard";
    map[SequenceCache::Type::FRAME_ENCODING_EVALUATION] = fs::path("frame_encoding") / "evaluation";
    return map;
}();


void SequenceCache::setPath(const std::filesystem::path& cachePath)
{
    _cachePath = cachePath;

    // Create sequence cache path directories in case they don't already exist
    for (const auto& [type, subPath] : typeSubPath)
        fs::create_directories(_cachePath / subPath);
}

uint64_t SequenceCache::nAvailableSequences(SequenceCache::Type cacheType) const
{
    uint64_t n = 0;
    auto iter = fs::directory_iterator(_cachePath / typeSubPath[cacheType]);
    for (const auto& s : iter)
        ++n;
    return n;
}

void SequenceCache::recordEntry(SequenceCache::Type cacheType, int batchId, int frameId, const Image<uint8_t>& image)
{
    newRecordSequenceName(cacheType);

    std::stringstream batchEntryIdSs, frameIdSs;
    batchEntryIdSs << std::setw(5) << std::setfill('0') << batchId;
    frameIdSs << std::setw(5) << std::setfill('0') << frameId << ".png";

    fs::path frameFilename = fs::path("temp") / _recordSequenceNames[cacheType] / batchEntryIdSs.str() /
        frameIdSs.str();

    fs::create_directories(frameFilename.parent_path());
    writeImageToFile(image, frameFilename);
}

void SequenceCache::finishRecord(Type cacheType)
{
    printf("Cache sequence record finished. Moving to %s\n",
        (_cachePath / typeSubPath[cacheType] / _recordSequenceNames[cacheType]).c_str());

    // Move the recorded sequence to the correct destination in the sequence cache
    fs::rename(fs::path("temp") / _recordSequenceNames[cacheType],
        _cachePath / typeSubPath[cacheType] / _recordSequenceNames[cacheType]);

    _recordSequenceNames[cacheType].clear();
}

void SequenceCache::deleteOldest(SequenceCache::Type cacheType)
{
    auto sequencePaths = [&](){
        std::vector<fs::path> paths;
        for (const auto& s : fs::directory_iterator(_cachePath / typeSubPath[cacheType])) {
            paths.push_back(_cachePath / typeSubPath[cacheType] / s);
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }();

    printf("Removing %s\n", sequencePaths[0].c_str());
    fs::remove_all(sequencePaths[0]);
}

void SequenceCache::loadToStorage(
    SequenceStorage& sequenceStorage,
    SequenceCache::Type cacheType,
    int nSequences,
    int offset
) {
    static std::vector<float> frameYUVData(doot2::frameWidth*doot2::frameHeight*
    getImageFormatNChannels(ImageFormat::YUV)); // external buffer to allow mapping to tensor
    static Image<float> frameYUV(doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV, frameYUVData.data());
    static const std::vector<int64_t> frameShape{doot2::frameHeight, doot2::frameWidth,
        getImageFormatNChannels(ImageFormat::YUV)};

    auto sequencePaths = [&](){
        std::vector<fs::path> paths;
        for (const auto& s : fs::directory_iterator(_cachePath / typeSubPath[cacheType])) {
            paths.push_back(_cachePath / typeSubPath[cacheType] / s);
        }
        return paths;
    }();

    if (nSequences > sequencePaths.size())
        throw std::runtime_error("Number of sequences requested exceeds the n. of sequences cached\n");

    for (int i=0; i<doot2::sequenceLength; ++i) {
        auto& base = sequencePaths[(offset + i) % nSequences];
        std::stringstream frameFilename;
        frameFilename << std::setw(5) << std::setfill('0') << i << ".png";
        for (int b=0; b<doot2::batchSize; ++b) {
            std::stringstream batchEntryDir;
            batchEntryDir << std::setw(5) << std::setfill('0') << b;
            fs::path filename = base / batchEntryDir.str() / frameFilename.str();
            //printf("loading from %s\n", filename.c_str());

            auto frame = readImageFromFile<uint8_t>(filename);
            convertImage(frame, frameYUV, ImageFormat::YUV);
            sequenceStorage.getBatch<float>("frame", i)[b] = torch::from_blob(
                frameYUVData.data(), frameShape, torch::TensorOptions().device(torch::kCPU)
            );
        }
    }
}

void SequenceCache::newRecordSequenceName(Type cacheType)
{
    if (_recordSequenceNames[cacheType].empty()) {
        using namespace std::chrono;
        std::stringstream ss;
        auto now = system_clock::to_time_t(system_clock::now());
        ss << std::put_time(std::gmtime(&now), "%Y%m%dT%H%M%S");
        _recordSequenceNames[cacheType] = ss.str();
    }
}

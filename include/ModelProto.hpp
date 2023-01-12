//
// Project: DooT2
// File: ModelProto.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "FrameEncoder.hpp"
#include "FrameDecoder.hpp"
#include "FlowDecoder.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>


class SequenceStorage;


class ModelProto {
public:
    ModelProto();
    ModelProto(const ModelProto&) = delete;
    ModelProto(ModelProto&&) = delete;
    ModelProto& operator=(const ModelProto&) = delete;
    ModelProto& operator=(ModelProto&&) = delete;

    void train(SequenceStorage&& storage);
    void trainAsync(SequenceStorage&& storage);
    bool trainingFinished() const noexcept;
    void waitForTrainingFinish();

private:
    FrameEncoder        _frameEncoder;
    FrameDecoder        _frameDecoder;
    FlowDecoder         _flowDecoder;
    torch::optim::Adam  _optimizer;
    std::mutex          _trainingMutex;
    std::thread         _trainingThread;
    std::atomic_bool    _trainingFinished;
};

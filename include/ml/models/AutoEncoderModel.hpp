//
// Project: DooT2
// File: AutoEncoderModel.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/Model.hpp"
#include "ml/modules/FrameEncoder.hpp"
#include "ml/modules/FrameDecoder.hpp"
#include "ml/modules/FlowDecoder.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>


class SequenceStorage;


namespace ml {

class AutoEncoderModel final : public Model {
public:
    AutoEncoderModel();
    AutoEncoderModel(const AutoEncoderModel&) = delete;
    AutoEncoderModel(AutoEncoderModel&&) = delete;
    AutoEncoderModel& operator=(const AutoEncoderModel&) = delete;
    AutoEncoderModel& operator=(AutoEncoderModel&&) = delete;

    void infer(const TensorVector& input, TensorVector& output) override;

private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    FrameEncoder        _frameEncoder;
    FrameDecoder        _frameDecoder;
    FlowDecoder         _flowDecoder;
    torch::optim::AdamW _optimizer;
    std::mutex          _trainingMutex;
    std::thread         _trainingThread;
    std::atomic_bool    _trainingFinished;
    TimePoint           _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

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

#include "Model.hpp"
#include "FrameEncoder.hpp"
#include "FrameDecoder.hpp"
#include "FlowDecoder.hpp"

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>


class SequenceStorage;


class AutoEncoderModel final : public Model {
public:
    AutoEncoderModel();
    AutoEncoderModel(const AutoEncoderModel&) = delete;
    AutoEncoderModel(AutoEncoderModel&&) = delete;
    AutoEncoderModel& operator=(const AutoEncoderModel&) = delete;
    AutoEncoderModel& operator=(AutoEncoderModel&&) = delete;

    void infer(const TensorVector& input, TensorVector& output) override;

private:
    FrameEncoder        _frameEncoder;
    FrameDecoder        _frameDecoder;
    FlowDecoder         _flowDecoder;
    torch::optim::AdamW _optimizer;
    std::mutex          _trainingMutex;
    std::thread         _trainingThread;
    std::atomic_bool    _trainingFinished;

    void trainImpl(SequenceStorage& storage) override;
};
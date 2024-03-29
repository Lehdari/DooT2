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
#include <chrono>


class SequenceStorage;


namespace ml {

struct TrainingInfo;


class AutoEncoderModel final : public Model {
public:
    AutoEncoderModel();
    AutoEncoderModel(const AutoEncoderModel&) = delete;
    AutoEncoderModel(AutoEncoderModel&&) = delete;
    AutoEncoderModel& operator=(const AutoEncoderModel&) = delete;
    AutoEncoderModel& operator=(AutoEncoderModel&&) = delete;

    void init(const nlohmann::json& experimentConfig) override;
    void setTrainingInfo(TrainingInfo* trainingInfo) override;
    void save(const std::filesystem::path& subdir = "") override;
    void infer(const TensorVector& input, TensorVector& output) override;

private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    std::filesystem::path   _frameEncoderFilename;
    std::filesystem::path   _frameDecoderFilename;
    std::filesystem::path   _flowDecoderFilename;

    FrameEncoder            _frameEncoder;
    FrameDecoder            _frameDecoder;
    FlowDecoder             _flowDecoder;
    torch::optim::AdamW     _optimizer;
    TimePoint               _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

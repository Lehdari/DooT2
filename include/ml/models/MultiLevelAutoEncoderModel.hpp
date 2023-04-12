//
// Project: DooT2
// File: MultiLevelMultiLevelAutoEncoderModel.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/Model.hpp"
#include "ml/modules/FrameEncoder.hpp"
#include "ml/modules/MultiLevelFrameDecoder.hpp"

#include <chrono>


class SequenceStorage;


namespace ml {

struct TrainingInfo;


class MultiLevelAutoEncoderModel final : public Model {
public:
    MultiLevelAutoEncoderModel(nlohmann::json* experimentConfig);
    MultiLevelAutoEncoderModel(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel(MultiLevelAutoEncoderModel&&) = delete;
    MultiLevelAutoEncoderModel& operator=(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel& operator=(MultiLevelAutoEncoderModel&&) = delete;

    void setTrainingInfo(TrainingInfo* trainingInfo) override;
    void save() override;
    void infer(const TensorVector& input, TensorVector& output) override;

private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    // Configuration variables and hyperparameters
    std::filesystem::path   _frameEncoderFilename;
    std::filesystem::path   _frameDecoderFilename;
    int64_t                 _optimizationInterval; // for this many steps gradients will be accumulated before weight update

    FrameEncoder            _frameEncoder;
    MultiLevelFrameDecoder  _frameDecoder;
    torch::optim::AdamW     _optimizer;
    TimePoint               _trainingStartTime;

    double                  _lossLevel;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

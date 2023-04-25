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
#include "ml/modules/MultiLevelFrameEncoder.hpp"
#include "ml/modules/MultiLevelFrameDecoder.hpp"

#include <chrono>


class SequenceStorage;


namespace ml {

struct TrainingInfo;


class MultiLevelAutoEncoderModel final : public Model {
public:
    static nlohmann::json getDefaultModelConfig();

    MultiLevelAutoEncoderModel();
    MultiLevelAutoEncoderModel(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel(MultiLevelAutoEncoderModel&&) = delete;
    MultiLevelAutoEncoderModel& operator=(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel& operator=(MultiLevelAutoEncoderModel&&) = delete;

    void init(const nlohmann::json& experimentConfig) override;
    void setTrainingInfo(TrainingInfo* trainingInfo) override;
    void save() override;
    void infer(const TensorVector& input, TensorVector& output) override;

private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    // Configuration variables and hyperparameters
    std::filesystem::path   _frameEncoderFilename;
    std::filesystem::path   _frameDecoderFilename;
    double                  _optimizerLearningRate;
    double                  _optimizerBeta1;
    double                  _optimizerBeta2;
    double                  _optimizerEpsilon;
    double                  _optimizerWeightDecay;
    int64_t                 _nTrainingIterations;
    int64_t                 _optimizationInterval; // for this many steps gradients will be accumulated before weight update
    double                  _frameLossWeight;
    double                  _frameGradLossWeight;
    double                  _frameLaplacianLossWeight;
    bool                    _useEncodingMeanLoss;
    double                  _encodingMeanLossWeight;
    bool                    _useEncodingCodistanceLoss;
    double                  _encodingCodistanceLossWeight;
    bool                    _useCovarianceLoss;
    double                  _covarianceLossWeight;
    double                  _targetLoss; // when loss is under this value, lossLevel is increased

    double                  _lossLevel; // determines the resolution the encoder

    MultiLevelFrameEncoder  _frameEncoder;
    MultiLevelFrameDecoder  _frameDecoder;
    torch::optim::AdamW     _optimizer;
    TimePoint               _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

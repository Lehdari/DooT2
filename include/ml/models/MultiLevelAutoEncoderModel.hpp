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
#include "ml/modules/Discriminator.hpp"

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
    std::filesystem::path                   _frameEncoderFilename;
    std::filesystem::path                   _frameDecoderFilename;
    std::filesystem::path                   _discriminatorFilename;
    double                                  _optimizerLearningRate;
    double                                  _optimizerBeta1;
    double                                  _optimizerBeta2;
    double                                  _optimizerEpsilon;
    double                                  _optimizerWeightDecay;
    int64_t                                 _nTrainingCycles; // number of times the sequence collection will be cycled through in each training call
    int64_t                                 _virtualBatchSize; // this many frames will gradients be accumulated over before the optimization step. must be at least 2
    double                                  _frameLossWeight;
    double                                  _frameGradLossWeight;
    double                                  _frameLaplacianLossWeight;
    bool                                    _useEncodingMeanLoss;
    double                                  _encodingMeanLossWeight;
    bool                                    _useEncodingCodistanceLoss;
    double                                  _encodingCodistanceLossWeight;
    bool                                    _useCovarianceLoss;
    double                                  _covarianceLossWeight;
    bool                                    _useEncodingPrevDistanceLoss;
    double                                  _encodingPrevDistanceLossWeight;
    double                                  _targetLoss; // when loss is under this value, lossLevel is increased

    double                                  _lossLevel; // determines the resolution the encoder
    double                                  _batchPixelDiff; // estimate for average pixel difference between frames from different sequences

    MultiLevelFrameEncoder                  _frameEncoder;
    MultiLevelFrameDecoder                  _frameDecoder;
    Discriminator                           _discriminator;
    std::unique_ptr<torch::optim::AdamW>    _optimizer;
    torch::Device                           _device;
    TimePoint                               _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;

    void scaleDisplayImages(
        const torch::Tensor& orig0,
        const torch::Tensor& orig1,
        const torch::Tensor& orig2,
        const torch::Tensor& orig3,
        const torch::Tensor& orig4,
        const torch::Tensor& orig5,
        torch::Tensor& image0,
        torch::Tensor& image1,
        torch::Tensor& image2,
        torch::Tensor& image3,
        torch::Tensor& image4,
        torch::Tensor& image5,
        torch::DeviceType device
    );

    static torch::Tensor createRandomEncodingInterpolations(const torch::Tensor& enc, double extrapolation=0.2);
};

} // namespace ml

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
#include "ml/modules/EncodingDiscriminator.hpp"
#include "ml/MultiLevelImage.hpp"
#include "util/Types.hpp"

#include <chrono>


class SequenceStorage;


namespace ml {

struct TrainingInfo;


class MultiLevelAutoEncoderModel final : public Model {
public:
    static Json getDefaultModelConfig();

    MultiLevelAutoEncoderModel();
    MultiLevelAutoEncoderModel(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel(MultiLevelAutoEncoderModel&&) = delete;
    MultiLevelAutoEncoderModel& operator=(const MultiLevelAutoEncoderModel&) = delete;
    MultiLevelAutoEncoderModel& operator=(MultiLevelAutoEncoderModel&&) = delete;

    void init(const Json& experimentConfig) override;
    void setTrainingInfo(TrainingInfo* trainingInfo) override;
    void save(const std::filesystem::path& subdir = "") override;
    void infer(const TensorVector& input, TensorVector& output) override;

private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    // Configuration variables and hyperparameters
    std::filesystem::path                   _experimentRoot;
    std::filesystem::path                   _frameEncoderFilename;
    std::filesystem::path                   _frameDecoderFilename;
    std::filesystem::path                   _discriminatorFilename;
    std::filesystem::path                   _encodingDiscriminatorFilename;
    std::filesystem::path                   _frameClassifierFilename;
    std::filesystem::path                   _optimizerStateFilename;
    double                                  _optimizerLearningRate;
    double                                  _optimizerBeta1;
    double                                  _optimizerBeta2;
    double                                  _optimizerEpsilon;
    double                                  _optimizerWeightDecay;
    double                                  _optimizerLearningRateInitial;
    double                                  _optimizerBeta1Initial;
    double                                  _optimizerWeightDecayInitial;
    double                                  _optimizerLearningRateFinal;
    double                                  _optimizerBeta1Final;
    double                                  _optimizerWeightDecayFinal;
    int64_t                                 _warmupDuration;
    int64_t                                 _nTrainingCycles; // number of times the sequence collection will be cycled through in each training call
    int64_t                                 _virtualBatchSize; // this many frames will gradients be accumulated over before the optimization step. must be at least 2
    double                                  _frameLossWeight;
    double                                  _frameGradLossWeight;
    double                                  _frameLaplacianLossWeight;
    bool                                    _useFrameClassificationLoss;
    double                                  _frameClassificationLossWeight;
    bool                                    _useEncodingMeanLoss;
    double                                  _encodingMeanLossWeight;
    bool                                    _useEncodingDistributionLoss;
    double                                  _encodingDistributionLossWeight;
    bool                                    _useEncodingDistanceLoss;
    double                                  _encodingDistanceLossWeight;
    bool                                    _useEncodingCovarianceLoss;
    double                                  _encodingCovarianceLossWeight;
    bool                                    _useEncodingPrevDistanceLoss;
    double                                  _encodingPrevDistanceLossWeight;
    bool                                    _useEncodingDiscriminationLoss;
    double                                  _encodingDiscriminationLossWeight;
    bool                                    _useEncodingCircularLoss;
    double                                  _encodingCircularLossWeight;
    bool                                    _useDiscriminator;
    double                                  _discriminationLossWeight;
    int64_t                                 _discriminatorVirtualBatchSize;
    double                                  _targetReconstructionLoss; // when loss is under this value, lossLevel is increased

    // State parameters (will be written to state_params.json)
    int64_t                                 _trainingIteration;
    double                                  _lossLevel; // determines the resolution of the encoding
    double                                  _batchPixelDiff; // estimate for average pixel difference between frames from different sequences
    double                                  _batchEncDiff; // estimate for average encoding distance between frames from different sequences

    MultiLevelFrameEncoder                  _frameEncoder;
    MultiLevelFrameDecoder                  _frameDecoder;
    Discriminator                           _discriminator;
    EncodingDiscriminator                   _encodingDiscriminator; // distinguishes between priori encodings and frame encodings
    EncodingDiscriminator                   _frameClassifier; // classifies encoded frames between real and generated
    std::unique_ptr<torch::optim::AdamW>    _optimizer;
    std::unique_ptr<torch::optim::AdamW>    _discriminatorOptimizer;
    std::unique_ptr<torch::optim::AdamW>    _encodingDiscriminatorOptimizer;
    torch::Device                           _device;
    TimePoint                               _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;

    MultiLevelImage scaleSequences(const SequenceStorage& storage);

    static void scaleDisplayImages(const MultiLevelImage& orig, MultiLevelImage& image, torch::DeviceType device);

    static torch::Tensor createRandomEncodingInterpolations(const torch::Tensor& enc, double extrapolation=0.2);

    void updateTrainingParameters(int64_t nVirtualBatchesPerCycle);
};

} // namespace ml

#pragma once

#include "ActionConverter.hpp"
#include "RewardModel.hpp"

#include <torch/torch.h>

class SequenceStorage;

class RewardModelTrainer {
public:
    RewardModelTrainer();
    RewardModelTrainer(const RewardModelTrainer&) = delete;
    RewardModelTrainer(RewardModelTrainer&&) = delete;
    RewardModelTrainer& operator=(const RewardModelTrainer&) = delete;
    RewardModelTrainer& operator=(RewardModelTrainer&&) = delete;

    void train(SequenceStorage& storage);
private:
    RewardModel                 _rewardModel;
    torch::optim::Adam          _optimizer;
    float                       _learningRate{1e-3};
    ActionConverter<float>      _actionConverter;
};
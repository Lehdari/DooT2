#pragma once

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
    torch::optim::Adam  _optimizer;
}
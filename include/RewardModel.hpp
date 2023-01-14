#pragma once

#include <torch/torch.h>
#include <torch/nn/modules/rnn.h>

class RewardModelImpl : public torch::nn::Module {
public:
    RewardModelImpl();
    torch::Tensor forward(
        torch::Tensor encodings,
        torch::Tensor actions,
        torch::Tensor rewards);
private:
    const int64_t _inputSize;
    const int64_t _hiddenSize;
    torch::nn::LSTM             _lstm;
};
TORCH_MODULE(RewardModel);

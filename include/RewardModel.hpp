#pragma once

#include <torch/torch.h>

class RewardModelImpl : public torch::nn::Module {
public:
    RewardModelImpl();
    torch::Tensor forward(torch::Tensor x);
private:
// layers here
    torch::nn::Conv2d           _conv1;//tmp
};
TORCH_MODULE(RewardModel);

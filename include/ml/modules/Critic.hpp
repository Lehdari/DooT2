#pragma once

#include <torch/torch.h>

namespace ml {
class CriticImpl : public torch::nn::Module {
public:
    CriticImpl();
    torch::Tensor forward(torch::Tensor x);
private:
torch::nn::Linear   _lin1;
};
TORCH_MODULE(Critic);

} // namespace ml
#pragma once

#include <torch/torch.h>

namespace ml {
class ActorImpl : public torch::nn::Module {
public:
    ActorImpl();
    torch::Tensor forward(torch::Tensor x);
private:
torch::nn::Linear   _lin1;
};
TORCH_MODULE(Actor);

} // namespace ml
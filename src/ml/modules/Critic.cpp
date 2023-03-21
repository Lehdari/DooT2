#include "ml/modules/Critic.hpp"


using namespace ml;
using namespace torch;

/*
Critic:
- takes N features in
- returns 1 scalar out, "goodness"
*/


CriticImpl::CriticImpl() :
_lin1(nn::LinearOptions(2,2))
{
    register_module("lin1", _lin1);
}

torch::Tensor CriticImpl::forward(torch::Tensor x)
{
    return torch::leaky_relu(_lin1(x));
}

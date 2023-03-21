#include "ml/modules/Actor.hpp"


using namespace ml;
using namespace torch;

/*
Actor:
- takes N features in
- returns X action logits out
*/


ActorImpl::ActorImpl() :
_lin1(nn::LinearOptions(2,2))
{
    register_module("lin1", _lin1);
}

torch::Tensor ActorImpl::forward(torch::Tensor x)
{
    return torch::leaky_relu(_lin1(x));
}

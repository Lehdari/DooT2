#include "RewardModel.hpp"

using namespace torch;

RewardModelImpl::RewardModelImpl() :
_conv1          (nn::Conv2dOptions(4, 16, {4, 4}).stride({2, 2}).bias(false).padding(1)) /*tmp*/
{
    register_module("conv1", _conv1);
}

torch::Tensor RewardModelImpl::forward(torch::Tensor x)
{
    return _conv1(x); //tmp
}
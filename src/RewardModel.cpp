#include "RewardModel.hpp"

#include "Constants.hpp"

#include <tuple>

using namespace doot2;
using namespace torch;

RewardModelImpl::RewardModelImpl() :
    _inputSize(actionVectorLength + encodingLength),
    _hiddenSize(encodingLength + 1),
    _lstm(_inputSize, _hiddenSize)
{
    register_module("lstm", _lstm);
}

// Action at time step t takes us to state t+1 and gives reward t+1
torch::Tensor RewardModelImpl::forward(
    torch::Tensor encodings,    /*LBW*/
    torch::Tensor actions,      /*LBW*/
    torch::Tensor rewards       /*LB*/
) 
{
    // Input: encoding and action
    // Output: encoding and reward

    torch::Tensor input = torch::cat({encodings, actions}, 2);

    // (output), (h_n, c_n)
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> outputRaw = _lstm(input);
    torch::Tensor output = std::get<0>(outputRaw);

    // 64 16 2049 0
    // printf("Output sizes: %ld %ld %ld %ld\n", output.sizes()[0], output.sizes()[1], output.sizes()[2], output.sizes()[3]);

    return output;
}

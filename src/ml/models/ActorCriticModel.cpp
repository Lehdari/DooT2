#include "ml/models/ActorCriticModel.hpp"
#include "util/SequenceStorage.hpp"


// TEMP
#include "Constants.hpp"
#include <gvizdoom/DoomGame.hpp>
#include <filesystem>
#include <random>



using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;
using namespace std::chrono;


ActorCriticModel::ActorCriticModel() :
_optimizer(
    {_actor->parameters(), _critic->parameters()}, 
    torch::optim::AdamWOptions(1.0e-3).betas({0.9, 0.999}).weight_decay(0.001)
),
_trainingFinished   (true),
_trainingStartTime  (std::chrono::high_resolution_clock::now())
{

}

void ActorCriticModel::trainImpl(SequenceStorage& storage)
{
    _trainingFinished = true;
}

void ActorCriticModel::infer(const TensorVector& input, TensorVector& output)
{

}

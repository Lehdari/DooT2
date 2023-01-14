#include "RewardModelTrainer.hpp"

#include "Constants.hpp"
#include "SequenceStorage.hpp"

using namespace doot2;
using namespace gvizdoom;


RewardModelTrainer::RewardModelTrainer() :
    _optimizer(
        {_rewardModel->parameters()},
        torch::optim::AdamOptions(_learningRate).betas({0.9, 0.999})
    )
{
    // TODO load reward model from file if it is not found on the disk

    printf("Reward model constructor\n"); // tmp

    _actionConverter.setAngleIndex(0);
    _actionConverter.setKeyIndex(1, Action::Key::ACTION_FORWARD);
    _actionConverter.setKeyIndex(2, Action::Key::ACTION_BACK);
    _actionConverter.setKeyIndex(3, Action::Key::ACTION_LEFT);
    _actionConverter.setKeyIndex(4, Action::Key::ACTION_RIGHT);
    _actionConverter.setKeyIndex(5, Action::Key::ACTION_USE);

}

void RewardModelTrainer::train(SequenceStorage& storage)
{
    torch::Tensor encodings = storage.mapEncodingData(); // LBW
    
    torch::Tensor actions(torch::zeros(
        {static_cast<uint32_t>(storage.settings().length),
        storage.settings().batchSize,
        actionVectorLength})); // LBW where W=6
    
    torch::Tensor rewards = storage.mapRewards().toType(torch::kF32); // LB

    auto* actionsPtr = actions.data_ptr<float>();

    for (size_t t = 0; t < storage.settings().length; ++t) {
        for (size_t bi = 0; bi < storage.settings().batchSize; ++bi) {
            auto batch = storage[bi];
            auto action = batch.actions[t];
            auto actionVector = _actionConverter(action, actionVectorLength);

            std::copy(actionVector.begin(), actionVector.end(),
                actionsPtr + t*storage.settings().batchSize + bi);

        }
    }
    
#if 0
    // Temp: perform one training step
    {
        _rewardModel->zero_grad();
        
        torch::Tensor y = _rewardModel->forward();
        torch::Tensor loss = torch::l2_loss(y,target);
        loss.backward();
        _optimizer.step();
    }
#endif
}

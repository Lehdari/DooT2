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
    using namespace torch::indexing;
    torch::Tensor encodings = storage.mapEncodingData(); // LBW
    torch::Tensor encodingsShifted{torch::zeros({storage.settings().length, storage.settings().batchSize, encodingLength})};
    encodingsShifted.index({Slice(1, None), Slice(), Slice()}) = encodings.index({Slice(None, -1), Slice(), Slice()});


    torch::Tensor actions(torch::zeros(
        {static_cast<uint32_t>(storage.settings().length),
        storage.settings().batchSize,
        actionVectorLength})); // LBW where W=6
    
    torch::Tensor rewards = storage.mapRewards().toType(torch::kF32); // LB
    rewards = torch::unsqueeze(rewards, 2);


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
    
    // Temp: perform one training step
    {
        _rewardModel->zero_grad();
        
        // Size: seqLen x batchSize x (encodingLen + 1)
        torch::Tensor y = _rewardModel->forward(encodings, actions, rewards);
        
        torch::Tensor yEncodings = y.index({Slice(), Slice(), Slice(None, -1)});
        torch::Tensor yRewards = y.index({Slice(), Slice(), Slice(-1)});

        printTensor(yEncodings, "yenc");
        printTensor(encodings, "enc");

        printTensor(yRewards, "yrew");
        printTensor(rewards, "rew");

        // yenc: 64 16 2048 0
        // enc: 64 16 2048 0

        // yrew: 64 16 1 0
        // rew: 64 16 0 0

        auto lossEnc = torch::mse_loss(yEncodings, encodings);
        printTensor(lossEnc, "lossEnc");
        auto lossReward = torch::mse_loss(yRewards, rewards);
        printTensor(lossReward, "lossReward");

        auto loss = lossEnc + lossReward;
        
        printf("Loss: %.5f + %.5f = %.5f\n",
            lossEnc.item<float>(),
            lossReward.item<float>(),
            loss.item<float>());

        loss.backward();
        _optimizer.step();
    }
}

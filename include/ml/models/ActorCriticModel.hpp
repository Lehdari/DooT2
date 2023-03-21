#pragma once

#include "ml/Model.hpp"
#include "ml/modules/Actor.hpp"
#include "ml/modules/Critic.hpp"

// Maybe needed, included for fun
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>


class SequenceStorage;


namespace ml {

class ActorCriticModel final : public Model {
public:
    ActorCriticModel();
    ActorCriticModel(const ActorCriticModel&) = delete;
    ActorCriticModel(ActorCriticModel&&) = delete;
    ActorCriticModel& operator=(const ActorCriticModel&) = delete;
    ActorCriticModel& operator=(ActorCriticModel&&) = delete;

    void infer(const TensorVector& input, TensorVector& output) override;

    void reset() override {}
    
private:
    using TimePoint = decltype(std::chrono::high_resolution_clock::now());

    Actor               _actor;
    Critic              _critic;

    torch::optim::AdamW _optimizer;

    // TODO: idk if needed, we will see
    std::mutex          _trainingMutex;
    std::thread         _trainingThread;
    std::atomic_bool    _trainingFinished;
    TimePoint           _trainingStartTime;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml
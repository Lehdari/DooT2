//
// Project: DooT2
// File: RandomWalkerModel.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include "ml/Model.hpp"


class Heatmap;


namespace ml {

class RandomWalkerModel final : public Model {
public:
    RandomWalkerModel(Heatmap* heatmap=nullptr);
    RandomWalkerModel(const RandomWalkerModel&) = delete;
    RandomWalkerModel(RandomWalkerModel&&) = delete;
    RandomWalkerModel& operator=(const RandomWalkerModel&) = delete;
    RandomWalkerModel& operator=(RandomWalkerModel&&) = delete;

    void init(const nlohmann::json& experimentConfig) override;
    void reset() override;
    void infer(const TensorVector& input, TensorVector& output) override;

private:
    Heatmap*        _heatmap;
    torch::Tensor   _actionPrev;
    float           _heatmapValuePrev;
    float           _heatmapValueAbsMovingAverage;
    Vec2f           _playerPosPrev;
    float           _playerVelocityMovingAverage;

    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

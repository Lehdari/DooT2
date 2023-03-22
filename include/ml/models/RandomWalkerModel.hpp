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

#include "ml/Model.hpp"


namespace ml {

class RandomWalkerModel final : public Model {
public:
    RandomWalkerModel();
    RandomWalkerModel(const RandomWalkerModel&) = delete;
    RandomWalkerModel(RandomWalkerModel&&) = delete;
    RandomWalkerModel& operator=(const RandomWalkerModel&) = delete;
    RandomWalkerModel& operator=(RandomWalkerModel&&) = delete;

    void infer(const TensorVector& input, TensorVector& output) override;

private:
    void trainImpl(SequenceStorage& storage) override;
};

} // namespace ml

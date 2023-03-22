//
// Project: DooT2
// File: RandomWalkerModel.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/models/RandomWalkerModel.hpp"
#include "Constants.hpp"

#include <torch/torch.h>


using namespace ml;


RandomWalkerModel::RandomWalkerModel()
{
}

void RandomWalkerModel::infer(const TensorVector& input, TensorVector& output)
{
    output.resize(1);
    output[0] = torch::randn({doot2::actionVectorLength});
}

void RandomWalkerModel::trainImpl(SequenceStorage& storage)
{
}

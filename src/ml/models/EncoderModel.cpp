//
// Project: DooT2
// File: EncoderModel.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/models/EncoderModel.hpp"


using namespace ml;


EncoderModel::EncoderModel()
{
}

void EncoderModel::init(const nlohmann::json& experimentConfig)
{
}

void EncoderModel::infer(const TensorVector& input, TensorVector& output)
{
    output.resize(1);
    output[0] = _encoder->forward(input[0]);
}

void EncoderModel::trainImpl(SequenceStorage& storage)
{
}

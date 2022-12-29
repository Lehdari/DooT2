//
// Project: DooT2
// File: ModelProto.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "AutoEncoder.hpp"

#include <vector>
#include <memory>


class SequenceStorage;


class ModelProto {
public:
    ModelProto();
    ModelProto(const ModelProto&) = delete;
    ModelProto(ModelProto&&) = delete;
    ModelProto& operator=(const ModelProto&) = delete;
    ModelProto& operator=(ModelProto&&) = delete;

    void train(const SequenceStorage& storage);

private:
    AutoEncoder         _autoEncoder;
    torch::optim::Adam  _optimizer;
};

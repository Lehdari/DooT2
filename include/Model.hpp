//
// Project: DooT2
// File: Model.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Types.hpp"


class SequenceStorage;


class Model {
public:
    virtual void train(SequenceStorage& storage) = 0;
    virtual void infer(const TensorVector& input, TensorVector& output) = 0;
};

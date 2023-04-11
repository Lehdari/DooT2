//
// Project: DooT2
// File: AutoEncoderModel2.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "Model.hpp"
#include "FrameEncoder.hpp"
#include "FrameDecoder2.hpp"

#include <vector>
#include <memory>
#include <atomic>


class SequenceStorage;


class AutoEncoderModel2 final : public Model {
public:
    AutoEncoderModel2();
    AutoEncoderModel2(const AutoEncoderModel2&) = delete;
    AutoEncoderModel2(AutoEncoderModel2&&) = delete;
    AutoEncoderModel2& operator=(const AutoEncoderModel2&) = delete;
    AutoEncoderModel2& operator=(AutoEncoderModel2&&) = delete;

    void infer(const TensorVector& input, TensorVector& output) override;

private:
    FrameEncoder        _frameEncoder;
    FrameDecoder2       _frameDecoder;
    torch::optim::AdamW _optimizer;
    double              _frameLossSmooth;
    double              _skipLevel;

    void trainImpl(SequenceStorage& storage) override;
};

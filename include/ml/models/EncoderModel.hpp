//
// Project: DooT2
// File: EncoderModel.hpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ml/Model.hpp"
#include "ml/modules/FrameEncoder.hpp"

#include <nlohmann/json.hpp>


namespace ml {

    class EncoderModel final : public Model {
    public:
        EncoderModel(nlohmann::json* experimentConfig);
        EncoderModel(const EncoderModel&) = delete;
        EncoderModel(EncoderModel&&) = delete;
        EncoderModel& operator=(const EncoderModel&) = delete;
        EncoderModel& operator=(EncoderModel&&) = delete;

        void infer(const TensorVector& input, TensorVector& output) override;

    private:
        FrameEncoder    _encoder;

        void trainImpl(SequenceStorage& storage) override;
    };

} // namespace ml

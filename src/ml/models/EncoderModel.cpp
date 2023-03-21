//
// Project: DooT2
// File: EncoderModel.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <filesystem>
#include "Constants.hpp"
#include "ml/models/EncoderModel.hpp"


using namespace ml;
namespace fs = std::filesystem;
using namespace torch;


EncoderModel::EncoderModel()
{
    // Load frame encoder
    if (fs::exists(doot2::frameEncoderFilename)) {
        printf("Loading frame encoder model from %s\n", doot2::frameEncoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(doot2::frameEncoderFilename);
        _encoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame encoder model.\n", doot2::frameEncoderFilename.c_str()); // TODO logging
    }
}

void EncoderModel::infer(const TensorVector& input, TensorVector& output)
{
    output.resize(1);
    output[0] = _encoder->forward(input[0]);
}

void EncoderModel::trainImpl(SequenceStorage& storage)
{
}

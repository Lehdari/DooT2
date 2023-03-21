//
// Project: DooT2
// File: Trainer.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/Trainer.hpp"
#include "ml/Model.hpp"

#include "gvizdoom/DoomGame.hpp"

#include <opencv2/highgui.hpp>


using namespace ml;
using namespace gvizdoom;
namespace fs = std::filesystem;


Trainer::Trainer(
    Model* model,
    Model* agentModel,
    Model* encoderModel,
    uint32_t batchSizeIn,
    size_t sequenceLengthIn
) :
    _rnd                        (1507715517),
    _quit                       (false),
    _sequenceStorage            (batchSizeIn),
    _frame                      (Image<uint8_t>(
                                    DoomGame::instance().getScreenWidth(),
                                    DoomGame::instance().getScreenHeight(),
                                    ImageFormat::BGRA)),
    _frameId                    (0),
    _batchEntryId               (0),
    _newPatchReady              (false),
    _model                      (model),
    _agentModel                 (agentModel),
    _encoderModel               (encoderModel)
    // _modelEnc                   (modelEnc),
    // _modelAc                    (modelAc)
{    
    if (_agentModel == nullptr or _model == nullptr)
    {
        throw std::runtime_error("model must not be nullptr");

    // Setup sequence storage
    _sequenceStorage.addSequence<Action>("action", Action(Action::ACTION_NONE, 0));
    _sequenceStorage.addSequence<float>("frame", torch::zeros({
        doot2::frameHeight, doot2::frameWidth, getImageFormatNChannels(ImageFormat::YUV)}));
//    _sequenceStorage.addSequence<float>("encoding");
    _sequenceStorage.addSequence<double>("reward", 0.0);
    _sequenceStorage.resize(sequenceLengthIn);

    // Setup action converter
    _actionConverter.setAngleIndex(0);
    _actionConverter.setKeyIndex(1, Action::Key::ACTION_FORWARD);
    _actionConverter.setKeyIndex(2, Action::Key::ACTION_BACK);
    _actionConverter.setKeyIndex(3, Action::Key::ACTION_LEFT);
    _actionConverter.setKeyIndex(4, Action::Key::ACTION_RIGHT);
    _actionConverter.setKeyIndex(5, Action::Key::ACTION_USE);

    // Create output directories if they do not exist
    if (not fs::exists(doot2::modelsDirectory)) {
        printf("Default models directory does not exist. Trying to create the directory\n");
        if (not fs::create_directories(fs::path(doot2::modelsDirectory))) {
            printf("Could not create the directory for models. Expect a crash upon training\n");
        }
    }

}

Trainer::~Trainer()
{
}

void Trainer::loop()
{
    auto& doomGame = DoomGame::instance();

    size_t recordBeginFrameId = 768+_rnd()%512;
    size_t recordEndFrameId = recordBeginFrameId + _sequenceStorage.length();

    // Setup YUV frame
    std::vector<float> frameYUVData(doot2::frameWidth*doot2::frameHeight*
        getImageFormatNChannels(ImageFormat::YUV)); // external buffer to allow mapping to tensor
    Image<float> frameYUV(doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV, frameYUVData.data());
    std::vector<int64_t> frameShape{doot2::frameHeight, doot2::frameWidth,
        getImageFormatNChannels(ImageFormat::YUV)};
    {   // Copy and convert the first frame
        auto frameHandle = _frame.write();
        frameHandle->copyFrom(doomGame.getPixelsBGRA());
        convertImage(*frameHandle, frameYUV, ImageFormat::YUV);
    }

    // Setup model I/O tensor vectors
    TensorVector frameTV(1); // just hosts the converted input frame
    TensorVector encodingTV(1); // frame converted into an encoding (output of the encoding model)
    TensorVector actionTV(1); // action output produced by the agent model

    while (!_quit) {
        // Map the frame into a tensor
        frameTV[0] = torch::from_blob(frameYUVData.data(),
            { 1 /*batch size*/, doot2::frameWidth, doot2::frameHeight,
            getImageFormatNChannels(ImageFormat::YUV) },
            torch::TensorOptions().device(torch::kCPU)
        );

        // Run encoder and agent model inference
        if (_encoderModel == nullptr) {
            // no encoder model in use, input raw frames to the model
            _agentModel->infer(frameTV, actionTV);
        }
        else {
            _encoderModel->infer(frameTV, encodingTV);
            _agentModel->infer(encodingTV, actionTV);
        }

        // Convert agent model output to an action for this timestep
        auto action = _actionConverter(actionTV[0]);

        // Update the game state, restart if required
        if (_frameId >= recordEndFrameId || doomGame.update(action)) {
            nextMap();
            recordBeginFrameId = 768+_rnd()%512;
            recordEndFrameId = recordBeginFrameId + _sequenceStorage.length();
            continue;
        }

        // Copy the DoomGame frame to local storage
        {
            auto frameHandle = _frame.write();
            frameHandle->copyFrom(doomGame.getPixelsBGRA());
        }

        // Record
        if (_frameId >= recordBeginFrameId) {
            auto recordFrameId = _frameId - recordBeginFrameId;
            _sequenceStorage.getBatch<Action>("action", recordFrameId)[_batchEntryId] = action;
            {   // convert the frame
                auto frameHandle = _frame.read();
                convertImage(*frameHandle, frameYUV, ImageFormat::YUV);
                _sequenceStorage.getBatch<float>("frame", recordFrameId)[_batchEntryId] = torch::from_blob(
                    frameYUVData.data(), frameShape, torch::TensorOptions().device(torch::kCPU)
                );
            }
            _sequenceStorage.getBatch<double>("reward", recordFrameId)[_batchEntryId] = 0.0; // TODO no rewards for now
        }

        // Train
        if (_newPatchReady) {
            _model->waitForTrainingFinished();
            // quit() might've been called in the meanwhile
            if (_quit) break;

            printf("Training...\n");
            _model->trainAsync(_sequenceStorage);
            _newPatchReady = false;
        }

        ++_frameId;
    }

    _model->waitForTrainingFinished();
}

void Trainer::quit()
{
    _quit = true;
}

const SingleBuffer<Image<uint8_t>>::ReadHandle Trainer::getFrameReadHandle()
{
    return _frame.read();
}

void Trainer::nextMap()
{
    auto& doomGame = DoomGame::instance();

    _frameId = 0;
    if (++_batchEntryId >= _sequenceStorage.batchSize()) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    newGameConfig.map = _rnd()%29 + 1;
    doomGame.restart(newGameConfig);

    _model->reset();
    _agentModel->reset();
    if (_encoderModel != nullptr)
        _encoderModel->reset();
}

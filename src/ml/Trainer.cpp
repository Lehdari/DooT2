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
#include "Constants.hpp"
#include "ml/Model.hpp"

#include "gvizdoom/DoomGame.hpp"

#include <opencv2/highgui.hpp>


using namespace ml;
using namespace gvizdoom;
namespace fs = std::filesystem;


Trainer::Trainer(Model* model, Model* agentModel, uint32_t batchSizeIn, size_t sequenceLengthIn) :
    _rnd                        (1507715517),
    _quit                       (false),
    _sequenceStorage            (SequenceStorage::Settings{batchSizeIn, sequenceLengthIn,
                                    true, false,
                                    doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV,
                                    doot2::encodingLength}),
    _frame                      (Image<uint8_t>(
                                    DoomGame::instance().getScreenWidth(),
                                    DoomGame::instance().getScreenHeight(),
                                    ImageFormat::BGRA)),
    _frameId                    (0),
    _batchEntryId               (0),
    _newPatchReady              (false),
    _model                      (model),
    _agentModel                 (agentModel)
{
    if (_model == nullptr)
        throw std::runtime_error("model must not be nullptr");

    auto& doomGame = DoomGame::instance();

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
    size_t recordEndFrameId = recordBeginFrameId + _sequenceStorage.settings().length;

    TensorVector dummyAgentInput; // TODO replace with actual agent input
    TensorVector agentOutput;

    while (!_quit) {
        // Action for this timestep
        _agentModel->infer(dummyAgentInput, agentOutput);
        auto action = _actionConverter(agentOutput[0]);

        // Update the game state, restart if required
        if (_frameId >= recordEndFrameId || doomGame.update(action)) {
            nextMap();
            recordBeginFrameId = 768+_rnd()%512;
            recordEndFrameId = recordBeginFrameId + _sequenceStorage.settings().length;
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
            auto batch = _sequenceStorage[recordFrameId];
            batch.actions[_batchEntryId] = action;
            {   // convert the frame
                auto frameHandle = _frame.read();
                convertImage(*frameHandle, batch.frames[_batchEntryId], ImageFormat::YUV);
            }
            batch.rewards[_batchEntryId] = 0.0; // TODO no rewards for now
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
    _model->abortTraining();
}

const SingleBuffer<Image<uint8_t>>::ReadHandle Trainer::getFrameReadHandle()
{
    return _frame.read();
}

void Trainer::nextMap()
{
    auto& doomGame = DoomGame::instance();

    _frameId = 0;
    if (++_batchEntryId >= _sequenceStorage.settings().batchSize) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    newGameConfig.map = _rnd()%29 + 1;

    doomGame.restart(newGameConfig);
}

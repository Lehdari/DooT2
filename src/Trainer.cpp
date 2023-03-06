//
// Project: DooT2
// File: Trainer.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "Trainer.hpp"
#include "Constants.hpp"
#include "Model.hpp"

#include "gvizdoom/DoomGame.hpp"

#include <opencv2/highgui.hpp>


using namespace gvizdoom;
namespace fs = std::filesystem;


Trainer::Trainer(Model* model, uint32_t batchSizeIn, size_t sequenceLengthIn) :
    _rnd                        (1507715517),
    _quit                       (false),
    _heatmapActionModule        (HeatmapActionModule::Settings{256, 32.0f}),
    _doorTraversalActionModule  (false),
    _sequenceStorage            (SequenceStorage::Settings{batchSizeIn, sequenceLengthIn,
                                    true, false,
                                    doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV,
                                    doot2::encodingLength}),
    _positionPlot               (1024, 1024, CV_32FC3, cv::Scalar(0.0f)),
    _initPlayerPos              (0.0f, 0.0f),
    _frameId                    (0),
    _batchEntryId               (0),
    _newPatchReady              (false),
    _model                      (model)
{
    if (_model == nullptr)
        throw std::runtime_error("model must not be nullptr");

    auto& doomGame = DoomGame::instance();

    // Setup ActionManager
    _actionManager.addModule(&_doorTraversalActionModule);
    _actionManager.addModule(&_heatmapActionModule);

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

void Trainer::quit()
{
    _quit = true;
    _model->abortTraining();
}

void Trainer::loop()
{
    auto& doomGame = DoomGame::instance();

    Vec2f playerPosScreen(0.0f, 0.0f);

    size_t recordBeginFrameId = 768+_rnd()%512;
    size_t recordEndFrameId = recordBeginFrameId + _sequenceStorage.settings().length;

    while (!_quit) {
        // Player position relative to the starting position (_initPlayerPos)
        static Vec2f playerPosRelative(0.0f, 0.0f);

        // Action for this timestep
        auto action = _actionManager(
            {_frameId, playerPosRelative}
        );

        // Update the game state, restart if required
        if (_frameId >= recordEndFrameId || doomGame.update(action)) {
            nextMap();
            recordBeginFrameId = 768+_rnd()%512;
            recordEndFrameId = recordBeginFrameId + _sequenceStorage.settings().length;
            continue;
        }

        // Player position is undefined before the first update, so _initPlayerPos has to be
        // defined here. Also apply the exit position priori to the heatmap action module.
        if (_frameId == 0) {
            _initPlayerPos = doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0);
            Vec2f exitPos = doomGame.getGameState<GameState::ExitPos>() - _initPlayerPos;
            exitPos(1) *= -1.0f;

            _heatmapActionModule.applyExitPositionPriori(exitPos);
        }

        // Record
        if (_frameId >= recordBeginFrameId) {
            auto recordFrameId = _frameId - recordBeginFrameId;
            auto batch = _sequenceStorage[recordFrameId];
            batch.actions[_batchEntryId] = action;
            Image<uint8_t> frame(doomGame.getScreenWidth(), doomGame.getScreenHeight(), ImageFormat::BGRA);
            frame.copyFrom(doomGame.getPixelsBGRA());
            convertImage(frame, batch.frames[_batchEntryId], ImageFormat::YUV);
            batch.rewards[_batchEntryId] = 0.0; // TODO no rewards for now
        }

        // Update relative player position
        playerPosRelative(0) = doomGame.getGameState<GameState::PlayerPos>()(0) - _initPlayerPos(0);
        playerPosRelative(1) = _initPlayerPos(1) - doomGame.getGameState<GameState::PlayerPos>()(1); // invert y

        // Update heatmap
        _heatmapActionModule.addGaussianSample(playerPosRelative, 1.0f, 100.0f);
        _heatmapActionModule.refreshNormalization();

        // Update position plot
        playerPosScreen = playerPosRelative * 0.125f;
        if (playerPosScreen(0) >= -512.0f && playerPosScreen(1) >= -512.0f &&
            playerPosScreen(0) < 512.0f && playerPosScreen(1) < 512.0f) {
            if (_heatmapActionModule.getDiff() > 0.0f)
                _positionPlot.at<Vec3f>((int)playerPosScreen(1)+512, (int)playerPosScreen(0)+512)(1) = 1.0f;
            else
                _positionPlot.at<Vec3f>((int)playerPosScreen(1)+512, (int)playerPosScreen(0)+512)(0) = 1.0f;
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

void Trainer::nextMap()
{
    auto& doomGame = DoomGame::instance();

    _actionManager.reset();
    _positionPlot *= 0.0f;

    _frameId = 0;
    if (++_batchEntryId >= _sequenceStorage.settings().batchSize) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    newGameConfig.map = _rnd()%29 + 1;

    doomGame.restart(newGameConfig);
}

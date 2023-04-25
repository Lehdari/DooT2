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
#include "ml/Models.hpp"
#include "ml/ModelTypeUtils.hpp"
#include "gui/State.hpp"
#include "util/ExperimentName.hpp"

#include "gvizdoom/DoomGame.hpp"
#include <opencv2/highgui.hpp>


using namespace ml;
using namespace gvizdoom;
namespace fs = std::filesystem;


Trainer::Trainer(
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
    _batchEntryId               (0),
    _newPatchReady              (false),
    _model                      (nullptr),
    _agentModel                 (agentModel),
    _encoderModel               (encoderModel),
    _playerDistanceThreshold    (_rnd()%768)
{

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
}

Trainer::~Trainer()
{
}

void Trainer::loop()
{
    if (_model == nullptr)
        throw std::runtime_error("Model to be trained must not be nullptr (did you forget to call configureExperiment()?)");
    if (_agentModel == nullptr)
        throw std::runtime_error("Agent model must not be nullptr");

    setupExperiment();

    auto& doomGame = DoomGame::instance();
    _playerInitPos = doomGame.getGameState<GameState::PlayerPos>();

    bool recording = false;
    int nRecordedFrames = 0;

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

    _quit = false;
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
        if (nRecordedFrames >= _sequenceStorage.length() || doomGame.update(action)) {
            nextMap(_batchEntryId+1);
            nRecordedFrames = 0;
            recording = false;
            continue;
        }

        // Copy the DoomGame frame to local storage
        {
            auto frameHandle = _frame.write();
            frameHandle->copyFrom(doomGame.getPixelsBGRA());
        }

        // Record
        if (recording) {
            _sequenceStorage.getBatch<Action>("action", nRecordedFrames)[_batchEntryId] = action;
            {   // convert the frame
                auto frameHandle = _frame.read();
                convertImage(*frameHandle, frameYUV, ImageFormat::YUV);
                _sequenceStorage.getBatch<float>("frame", nRecordedFrames)[_batchEntryId] = torch::from_blob(
                    frameYUVData.data(), frameShape, torch::TensorOptions().device(torch::kCPU)
                );
            }
            _sequenceStorage.getBatch<double>("reward", nRecordedFrames)[_batchEntryId] = 0.0; // TODO no rewards for now
            ++nRecordedFrames;
        }
        else if (startRecording()) {
            recording = true;
        }

        // Train
        if (_newPatchReady) {
            _model->waitForTrainingFinished();
            // quit() might've been called in the meanwhile
            if (_quit) break;

            printf("Training...\n");
            _model->trainAsync(_sequenceStorage);
            _visitedMaps.clear();
            _newPatchReady = false;
        }
    }

    _model->waitForTrainingFinished();
    nextMap();
}

void Trainer::quit()
{
    _quit = true;
    _model->abortTraining();
}

void Trainer::configureExperiment(nlohmann::json&& experimentConfig)
{
    // Check for mandatory entries
    if (!experimentConfig.contains("experiment_name"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"experiment_name\"");
    if (!experimentConfig.contains("model_type"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"model_type\"");
    if (!experimentConfig.contains("model_config"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"model_config\"");

    _experimentConfig = std::move(experimentConfig);

    // Reset training info
    _trainingInfo.reset(); // TODO load past data if basis experiment is used

    // delete the previous model
    _model.reset();

    // instantiate new model of desired type using the config above
    ml::modelTypeNameCallback(_experimentConfig["model_type"], [&]<typename T_Model>(){
        _model = std::make_unique<T_Model>();
    });

    _model->setTrainingInfo(&_trainingInfo);
}

void Trainer::saveExperiment()
{
    printf("Saving the experiment\n"); // TODO logging

    createExperimentDirectories();

    fs::path experimentDir = doot2::experimentsDirectory / _experimentConfig["experiment_root"];

    // Save the experiment config
    {
        std::ofstream experimentConfigFile(experimentDir / "experiment_config.json");
        experimentConfigFile << std::setw(4) << _experimentConfig;
    }

    // Save the trained model
    _model->save();

    // Save the time series data
    {
        std::ofstream timeSeriesFile(experimentDir / "time_series.json");
        timeSeriesFile << _trainingInfo.timeSeries.read()->toJson();
    }
}

Model* Trainer::getModel()
{
    return _model.get();
}

TrainingInfo* Trainer::getTrainingInfo()
{
    return &_trainingInfo;
}

nlohmann::json* Trainer::getExperimentConfig()
{
    return &_experimentConfig;
}

const SingleBuffer<Image<uint8_t>>::ReadHandle Trainer::getFrameReadHandle()
{
    return _frame.read();
}

bool Trainer::startRecording()
{
    auto& doomGame = DoomGame::instance();
    float playerDistanceFromStart = (doomGame.getGameState<GameState::PlayerPos>() - _playerInitPos).norm();
    return playerDistanceFromStart > _playerDistanceThreshold && _rnd()%256 == 0;
}

void Trainer::nextMap(size_t newBatchEntryId)
{
    auto& doomGame = DoomGame::instance();

    _newPatchReady = false;
    _batchEntryId = newBatchEntryId;
    if (_batchEntryId >= _sequenceStorage.batchSize()) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    do {
        newGameConfig.map = _rnd()%29 + 1;
    } while (_visitedMaps.contains(newGameConfig.map));
    _visitedMaps.insert(newGameConfig.map);
    doomGame.restart(newGameConfig);
    doomGame.update(gvizdoom::Action()); // one update required for init position
    _playerInitPos = doomGame.getGameState<GameState::PlayerPos>();
    _playerDistanceThreshold = _rnd()%768;

    _model->reset();
    _agentModel->reset();
    if (_encoderModel != nullptr)
        _encoderModel->reset();
}

void Trainer::setupExperiment()
{
    // Format the experiment name
    _experimentConfig["experiment_name"] = formatExperimentName(_experimentConfig["experiment_name"],
        _experimentConfig["model_config"]);

    if (!_experimentConfig.contains("experiment_root")) {
        printf("INFO: No \"experiment_root\" defined. Using experiment_name: \"%s\"\n",
            _experimentConfig["experiment_name"].get<std::string>().c_str());
        _experimentConfig["experiment_root"] = _experimentConfig["experiment_name"];
    }

    createExperimentDirectories();

    _model->init(_experimentConfig);
}

void Trainer::createExperimentDirectories() const
{
    fs::path experimentDir = doot2::experimentsDirectory / _experimentConfig["experiment_root"];
    if (!fs::exists(experimentDir)) {
        printf("Creating experiment directory \"%s\"\n", experimentDir.c_str()); // TODO logging
        if (!fs::create_directories(experimentDir))
            throw std::runtime_error("Could not create the directory \""+experimentDir.string()+"\"");
    }
}

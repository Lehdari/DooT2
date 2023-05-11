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
#include "util/ExperimentUtils.hpp"

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
    _finished                   (true),
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

    auto& doomGame = DoomGame::instance();

    // Pick random first map
    _visitedMaps.clear();
    nextMap();

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

    bool recording = false;
    bool evaluating = false;
    int nRecordedFrames = 0;
    int epoch = 0;
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
            nextMap(_batchEntryId+1, evaluating);
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

        // Train or evaluate
        if (_newPatchReady) {
            _model->waitForTrainingFinished();
            // quit() might've been called in the meanwhile
            if (_quit) break;

            if (evaluating) {
                evaluateModel();
                evaluating = false;
                nextMap();
            }
            else { // training
                _model->trainAsync(_sequenceStorage);
                _visitedMaps.clear();
                _newPatchReady = false;
                ++epoch;

                // Check for evaluation epoch
                if (_experimentConfig.contains("evaluation_interval")) {
                    assert(_experimentConfig.contains("pwad_filenames_evaluation"));
                    if (epoch % _experimentConfig["evaluation_interval"].get<int>() == 0) {
                        // evaluation interval epoch hit
                        evaluating = true;
                    }
                }
            }
        }

        // number of epochs training termination condition
        if (_experimentConfig.contains("n_training_epochs") &&
            epoch >= _experimentConfig["n_training_epochs"].get<int>()) {
            printf("INFO: n_training_epochs reached: %d / %d, terminating experiment...\n", epoch,
                _experimentConfig["n_training_epochs"].get<int>()); // TODO logging
            break;
        }
    }

    _model->waitForTrainingFinished();
    nextMap();
    _finished = true;
}

void Trainer::quit()
{
    _quit = true;
    _model->abortTraining();
}

bool Trainer::isFinished()
{
    return _finished;
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

    std::string previousModelType;
    if (_experimentConfig.contains("model_type"))
        previousModelType = _experimentConfig["model_type"];

    _experimentConfig = std::move(experimentConfig);

    // Reset training info, load base experiment data if one is specified
    loadBaseExperimentTrainingInfo();

    // Check if the model type has changed, reinstantiate the model in that case
    if (previousModelType != _experimentConfig["model_type"].get<std::string>()) {
        // delete the previous model
        _model.reset();

        // instantiate new model of desired type using the config above
        ml::modelTypeNameCallback(_experimentConfig["model_type"], [&]<typename T_Model>() {
            _model = std::make_unique<T_Model>();
        });
    }

    _model->setTrainingInfo(&_trainingInfo);
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
    loadBaseExperimentTrainingInfo();

    if (_experimentConfig.contains("evaluation_interval")) {
        // Add evaluation input/output images
        if (!_trainingInfo.images.contains("evaluation_input"))
            *_trainingInfo.images["evaluation_input"].write() = Image<float>(
                doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV);
        if (!_trainingInfo.images.contains("evaluation_output"))
            *_trainingInfo.images["evaluation_output"].write() = Image<float>(
                doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV);

        // Add evaluation time series
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("time", 0.0);
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("performanceAvg", 0.0);
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("performanceMin", 0.0);
    }

    _model->setTrainingInfo(&_trainingInfo);
    _model->init(_experimentConfig);
    _finished = false;
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
        std::ofstream timeSeriesFile(experimentDir / "training_time_series.json");
        timeSeriesFile << _trainingInfo.trainingTimeSeries.read()->toJson();
    }
    {
        std::ofstream timeSeriesFile(experimentDir / "evaluation_time_series.json");
        timeSeriesFile << _trainingInfo.evaluationTimeSeries.read()->toJson();
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

void Trainer::nextMap(size_t newBatchEntryId, bool evaluating)
{
    auto& doomGame = DoomGame::instance();

    _newPatchReady = false;
    _batchEntryId = newBatchEntryId;
    if (_batchEntryId >= _sequenceStorage.batchSize()) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    // Use training or evaluation map wads based on whether evaluation was requested
    std::vector<fs::path> wadFilenames;
    if (evaluating) {
        assert(_experimentConfig.contains("pwad_filenames_evaluation"));
        wadFilenames = _experimentConfig["pwad_filenames_evaluation"];
    }
    else {
        assert(_experimentConfig.contains("pwad_filenames_training"));
        wadFilenames = _experimentConfig["pwad_filenames_training"];
    }
    auto nWads = wadFilenames.size();

    // Pick a new map that hasn't been visited in this batch before and restart the game
    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    int newMapId = 0;
    do {
        auto wadId = _rnd() % nWads;
        newGameConfig.pwadFileNames = {wadFilenames[wadId]};
        newGameConfig.map = _rnd()%29 + 1;
        newMapId = (int)wadId*100 + newGameConfig.map;
    } while (_visitedMaps.contains(newMapId));
    _visitedMaps.insert(newMapId);
    doomGame.restart(newGameConfig);
    doomGame.update(gvizdoom::Action()); // one update required for init position
    _playerInitPos = doomGame.getGameState<GameState::PlayerPos>();
    _playerDistanceThreshold = _rnd()%768;

    // Reset models
    _model->reset();
    _agentModel->reset();
    if (_encoderModel != nullptr)
        _encoderModel->reset();
}

void Trainer::evaluateModel()
{
    TensorVector inputs(1);
    TensorVector outputs(1);

    double performanceAvg = 0.0;
    double performanceMin = std::numeric_limits<double>::max();
    int nSamples = 0;
    for (int i=0; i<doot2::batchSize; ++i) {
        auto* storageFrames = _sequenceStorage.getSequence<float>("frame");
        assert(storageFrames != nullptr);
        torch::Tensor pixelDataIn = storageFrames->tensor();
        for (int t=0; t<storageFrames->length(); ++t) {
            using namespace torch::indexing;
            inputs[0] = pixelDataIn.index({t, i, "..."}).unsqueeze(0).permute({0, 3, 1, 2});
            _model->infer(inputs, outputs);

            _trainingInfo.images["evaluation_input"].write()->copyFrom(
                inputs[0].permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo.images["evaluation_output"].write()->copyFrom(
                outputs[0].permute({0, 2, 3, 1}).contiguous().data_ptr<float>());

            // TODO introduce proper performance metric(s)
            auto loss = torch::mean(torch::abs(inputs[0]-outputs[0])).item<double>();
            double performance = 1.0 / loss; // use inverse of l1 loss for now

            performanceAvg += performance;
            performanceMin = std::min(performanceMin, performance);
            ++nSamples;
        }
    }

    performanceAvg /= nSamples;

    _trainingInfo.evaluationTimeSeries.write()->addEntries(
        "performanceAvg", performanceAvg,
        "performanceMin", performanceMin
    );
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

void Trainer::loadBaseExperimentTrainingInfo()
{
    _trainingInfo.reset();

    if (!_experimentConfig.contains("experiment_base_root"))
        return; // no base experiment specified, return

    // Load training time series
    fs::path trainingTimeSeriesPath = _experimentConfig["experiment_base_root"].get<fs::path>() / "training_time_series.json";
    if (fs::exists(trainingTimeSeriesPath)) {
        // TODO handle potential exceptions
        std::ifstream timeSeriesFile(trainingTimeSeriesPath);
        auto timeSeriesJson = nlohmann::json::parse(timeSeriesFile);
        _trainingInfo.trainingTimeSeries.write()->fromJson<double>(timeSeriesJson);
    }
    else {
        printf("WARNING: Base experiment specified in config but training time series data was not found (%s).\n",
            trainingTimeSeriesPath.c_str()); // TODO logging
        return;
    }

    // Load evaluation time series
    if (_experimentConfig.contains("evaluation_interval")) {
        fs::path evaluationTimeSeriesPath = _experimentConfig["experiment_base_root"].get<fs::path>() / "evaluation_time_series.json";
        if (fs::exists(evaluationTimeSeriesPath)) {
            // TODO handle potential exceptions
            std::ifstream timeSeriesFile(evaluationTimeSeriesPath);
            auto timeSeriesJson = nlohmann::json::parse(timeSeriesFile);
            _trainingInfo.evaluationTimeSeries.write()->fromJson<double>(timeSeriesJson);
        }
        else {
            printf("WARNING: Base experiment and evaluation interval specified in config but evaluation time series data was not found (%s).\n",
                evaluationTimeSeriesPath.c_str()); // TODO logging
            return;
        }
    }
}

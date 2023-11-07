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
#include "util/ImageUtils.hpp"

#include "gvizdoom/DoomGame.hpp"
#include <opencv2/highgui.hpp>


using namespace ml;
using namespace gvizdoom;
namespace fs = std::filesystem;
namespace tf = torch::nn::functional;


Trainer::Trainer(
    Model* agentModel,
    Model* encoderModel,
    uint32_t batchSizeIn,
    size_t sequenceLengthIn
) :
    _rnd                        (time(NULL)),
    _quit                       (false),
    _finished                   (true),
    _sequenceStorage            (batchSizeIn),
    _frameRGB                   (Image<uint8_t>(
                                 DoomGame::instance().getScreenWidth(),
                                 DoomGame::instance().getScreenHeight(),
                                 ImageFormat::BGRA)),
    _frameYUVData               (doot2::frameWidth*doot2::frameHeight*getImageFormatNChannels(ImageFormat::YUV)),
    _frameYUV                   (doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV, _frameYUVData.data()),
    _model                      (nullptr),
    _agentModel                 (agentModel),
    _encoderModel               (encoderModel),
    _playerDistanceThreshold    (_rnd()%768)
{

    // Setup sequence storage
    for (int i=0; i<2; ++i) {
        {
            auto storage = _sequenceStorage.write();
            storage->addSequence<Action>("action", Action(Action::ACTION_NONE, 0));
//            storage->addSequence<float>("frame", torch::zeros({
//                doot2::frameHeight, doot2::frameWidth, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame6", torch::zeros({
                doot2::frameHeight/2, doot2::frameWidth/2, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame5", torch::zeros({
                doot2::frameHeight/4, doot2::frameWidth/4, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame4", torch::zeros({
                doot2::frameHeight/8, doot2::frameWidth/8, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame3", torch::zeros({
                doot2::frameHeight/16, doot2::frameWidth/16, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame2", torch::zeros({
                doot2::frameHeight/32, doot2::frameWidth/32, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame1", torch::zeros({
                doot2::frameHeight/32, doot2::frameWidth/64, getImageFormatNChannels(ImageFormat::YUV)}));
            storage->addSequence<float>("frame0", torch::zeros({
                doot2::frameHeight/96, doot2::frameWidth/128, getImageFormatNChannels(ImageFormat::YUV)}));

            //    _sequenceStorage.addSequence<float>("encoding");
            storage->addSequence<double>("reward", 0.0);
            storage->resize(sequenceLengthIn);
        }
        _sequenceStorage.read(); // read swaps the buffers so that both buffers get initialized
    }

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

    // Pick random first map
    _visitedMaps.clear();
    nextMap();

    int epoch = 0;
    _quit = false;
    while (!_quit) {
        printf("Epoch %d\n", epoch); fflush(stdout);

        // Training
        {
            refreshSequenceStorage(epoch); // Load new sequences for training
            _model->waitForTrainingFinished();
            if (_quit) break; // quit() might've been called in the meanwhile
            _model->trainAsync(_sequenceStorage.read()); // Launch asynchronous training
            ++epoch;
        }

        // Run cache update (deletion of oldest sequences, recording of new ones etc.)
        updateCache(epoch);

        // Evaluation
        if (isEvaluationEpoch(epoch)) {
            refreshSequenceStorage(epoch, true); // Load new sequences for evaluation
            _model->waitForTrainingFinished();
            saveExperiment();
            evaluateModel();
        }

        // number of epochs training termination condition
        if (terminationEpochReached(epoch))
            break;

        {   // Wait for unpause signal in case the training has been paused
            std::unique_lock pauseLock(_pauseMutex);
            if (_paused)
                saveExperiment();
            _pauseCV.wait(pauseLock, [this]{ return !_paused || _quit; });
        }
    }

    _model->waitForTrainingFinished();
    nextMap();
    _finished = true;
}

void Trainer::quit()
{
    {
        std::lock_guard lock(_pauseMutex);
        _quit = true;
    }
    _pauseCV.notify_one();
    _model->abortTraining();
}

bool Trainer::isFinished() const noexcept
{
    return _finished;
}

void Trainer::pause()
{
    {
        std::lock_guard lock(_pauseMutex);
        _paused = true;
    }
    _pauseCV.notify_one();
}

void Trainer::unpause()
{
    {
        std::lock_guard lock(_pauseMutex);
        _paused = false;
    }
    _pauseCV.notify_one();
}

bool Trainer::isPaused() const noexcept
{
    return _paused;
}

void Trainer::configureExperiment(nlohmann::json&& experimentConfig)
{
    // Check for mandatory entries
    if (!experimentConfig.contains("experiment_name"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"experiment_name\"");
    if (!experimentConfig.contains("training_task"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"training_task\"");
    if (!experimentConfig.contains("model_type"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"model_type\"");
    if (!experimentConfig.contains("model_config"))
        throw std::runtime_error("Experiment config does not contain mandatory entry \"model_config\"");

    // Check task-specific entries
    if (experimentConfig["training_task"].get<int32_t>() == 0) {
        if (!experimentConfig.contains("use_sequence_cache"))
            throw std::runtime_error("Experiment config does not contain entry \"use_sequence_cache\"");
        if (experimentConfig["use_sequence_cache"].get<bool>()) { // using cache, check for parameters
            if (!experimentConfig.contains("sequence_cache_path"))
                throw std::runtime_error("Experiment config does not contain entry \"sequence_cache_path\"");
            if (!experimentConfig.contains("n_cached_sequences"))
                throw std::runtime_error("Experiment config does not contain entry \"n_cached_sequences\"");
        }
    }

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
        if (!_trainingInfo.images.contains("evaluation_encoding"))
            *_trainingInfo.images["evaluation_encoding"].write() = Image<float>(64*8, 32*8, ImageFormat::GRAY);
        if (!_trainingInfo.images.contains("evaluation_encoding_mask"))
            *_trainingInfo.images["evaluation_encoding_mask"].write() = Image<float>(64*8, 32*8, ImageFormat::GRAY);

        // Add evaluation time series
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("time", 0.0);
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("performanceAvg", 0.0);
        _trainingInfo.evaluationTimeSeries.write()->addSeries<double>("performanceMin", 0.0);
    }

    // Setup sequence cache
    if (_experimentConfig["use_sequence_cache"].get<bool>())
        _sequenceCache.setPath(_experimentConfig["sequence_cache_path"].get<fs::path>());

    _model->init(_experimentConfig);
    _model->setTrainingInfo(&_trainingInfo);
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
    return _frameRGB.read();
}

void Trainer::refreshSequenceStorage(int epoch, bool evaluation)
{
    auto storage = _sequenceStorage.write();

    if (_experimentConfig.contains("use_sequence_cache") &&
        _experimentConfig["use_sequence_cache"].get<bool>()) {
        // Using sequence cache
        if (evaluation) {
            // Evaluation sequences requested
            if (_sequenceCache.nAvailableSequences(SequenceCache::Type::FRAME_ENCODING_EVALUATION) < 1) {
                // Cache does not have enough sequences, record new ones until it does
                recordEvaluationSequences();
            }
//            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
//                "frame", 1, epoch);

            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame6", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame5", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame4", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame3", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame2", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame1", 1, epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                "frame0", 1, epoch);
        }
        else {
            // Training sequences requested
            if (_sequenceCache.nAvailableSequences(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL) <
                _experimentConfig["n_cached_sequences"]) {
                // Cache does not have enough sequences, record new ones until it does
                recordTrainingSequences();
            }
//            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
//                "frame", _experimentConfig["n_cached_sequences"].get<int>(), epoch);

            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame6", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame5", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame4", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame3", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame2", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame1", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
            _sequenceCache.loadFramesToStorage(*storage, SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                "frame0", _experimentConfig["n_cached_sequences"].get<int>(), epoch);
        }
    }
    else {
        // Not using sequence cache
        // TODO
    }
}

void Trainer::recordTrainingSequences()
{
    while (_sequenceCache.nAvailableSequences(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL) <
        _experimentConfig["n_cached_sequences"]) {
        _visitedMaps.clear();
        for (int b=0; b<doot2::batchSize;) {
            int mapId = nextMap(); // load new map from the training set
            // Run the game until recording is indicated to start
            while (!startRecording()) {
                if (gameStep()) // exit was reached prematurely, just load a new map for now
                    mapId = nextMap();
            }

            bool aborted = false;
            for (int i=0; i<doot2::sequenceLength; ++i) {
                if (gameStep()) { // exit was reached prematurely, start over for the sequence
                    aborted = true;
                    break;
                }

                // Record the frame
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame", b, i, *_frameRGB.read());
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame6", b, i, downscaleImage(*_frameRGB.read(), 2, 2));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame5", b, i, downscaleImage(*_frameRGB.read(), 4, 4));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame4", b, i, downscaleImage(*_frameRGB.read(), 8, 8));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame3", b, i, downscaleImage(*_frameRGB.read(), 16, 16));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame2", b, i, downscaleImage(*_frameRGB.read(), 32, 32));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame1", b, i, downscaleImage(*_frameRGB.read(), 64, 32));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL,
                    "frame0", b, i, downscaleImage(*_frameRGB.read(), 128, 96));
            }

            if (aborted)
                continue;

            _visitedMaps.insert(mapId);
            ++b;
        }

        // Sequence batch complete, transfer it to cache
        _sequenceCache.finishRecord(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL);
    }
}

void Trainer::recordEvaluationSequences()
{
    while (_sequenceCache.nAvailableSequences(SequenceCache::Type::FRAME_ENCODING_EVALUATION) < 1) {
        _visitedMaps.clear();
        for (int b=0; b<doot2::batchSize;) {
            int mapId = nextMap(true); // load new map from the training set
            // Run the game until recording is indicated to start
            while (!startRecording()) {
                if (gameStep()) // exit was reached prematurely, just load a new map for now
                    mapId = nextMap(true);
            }

            bool aborted = false;
            for (int i=0; i<doot2::sequenceLength; ++i) {
                if (gameStep()) { // exit was reached prematurely, start over for the sequence
                    aborted = true;
                    break;
                }

                // Record the frame
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame", b, i, *_frameRGB.read());
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame6", b, i, downscaleImage(*_frameRGB.read(), 2, 2));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame5", b, i, downscaleImage(*_frameRGB.read(), 4, 4));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame4", b, i, downscaleImage(*_frameRGB.read(), 8, 8));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame3", b, i, downscaleImage(*_frameRGB.read(), 16, 16));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame2", b, i, downscaleImage(*_frameRGB.read(), 32, 32));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame1", b, i, downscaleImage(*_frameRGB.read(), 64, 32));
                _sequenceCache.recordEntry(SequenceCache::Type::FRAME_ENCODING_EVALUATION,
                    "frame0", b, i, downscaleImage(*_frameRGB.read(), 128, 96));
            }

            if (aborted)
                continue;

            _visitedMaps.insert(mapId);
            ++b;
        }

        // Sequence batch complete, transfer it to cache
        _sequenceCache.finishRecord(SequenceCache::Type::FRAME_ENCODING_EVALUATION);
    }
}

bool Trainer::isEvaluationEpoch(int epoch) const
{
    if (_experimentConfig.contains("evaluation_interval") &&
        epoch % _experimentConfig["evaluation_interval"].get<int>() == 0) {
        // evaluation interval epoch hit
        return true;
    }

    return false;
}

bool Trainer::terminationEpochReached(int epoch) const
{
    if (_experimentConfig.contains("n_training_epochs") &&
        epoch >= _experimentConfig["n_training_epochs"].get<int>()) {
        printf("INFO: n_training_epochs reached: %d / %d, terminating experiment...\n", epoch,
            _experimentConfig["n_training_epochs"].get<int>()); // TODO logging
        return true;
    }

    return false;
}

void Trainer::updateCache(int epoch)
{
    if (_experimentConfig.contains("sequence_cache_training_record_interval") &&
        epoch % _experimentConfig["sequence_cache_training_record_interval"].get<int>() == 0)
    {
        // Delete the oldest sequence batch and record a new one
        _sequenceCache.deleteOldest(SequenceCache::Type::FRAME_ENCODING_TRAINING_NORMAL);
        recordTrainingSequences();
    }
}

bool Trainer::gameStep()
{
    auto& doomGame = DoomGame::instance();

    // Setup YUV frame
    static const std::vector<int64_t> frameShape{doot2::frameHeight, doot2::frameWidth,
        getImageFormatNChannels(ImageFormat::YUV)};
    {   // Copy and convert the first frame
        auto frameHandle = _frameRGB.write();
        frameHandle->copyFrom(doomGame.getPixelsBGRA());
        convertImage(*frameHandle, _frameYUV, ImageFormat::YUV);
    }

    // Setup model I/O tensor vectors
    static TensorVector frameTV(1); // just hosts the converted input frame
    static TensorVector encodingTV(1); // frame converted into an encoding (output of the encoding model)
    static TensorVector actionTV(1); // action output produced by the agent model

    // Map the frame into a tensor
    frameTV[0] = torch::from_blob(_frameYUVData.data(),
        { 1 /*batch size*/, doot2::frameWidth, doot2::frameHeight, getImageFormatNChannels(ImageFormat::YUV) },
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
    bool exitReached = doomGame.update(action);

    // Copy the DoomGame frame to local storage
    {
        auto frameHandle = _frameRGB.write();
        frameHandle->copyFrom(doomGame.getPixelsBGRA());
    }

    return exitReached;
}

bool Trainer::startRecording()
{
    auto& doomGame = DoomGame::instance();
    float playerDistanceFromStart = (doomGame.getGameState<GameState::PlayerPos>() - _playerInitPos).norm();
    return playerDistanceFromStart > _playerDistanceThreshold && _rnd()%256 == 0;
}

int Trainer::nextMap(bool evaluation)
{
    auto& doomGame = DoomGame::instance();

    // Use training or evaluation map wads based on whether evaluation was requested
    std::vector<fs::path> wadFilenames;
    if (evaluation) {
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
    doomGame.restart(newGameConfig);
    doomGame.update(gvizdoom::Action()); // one update required for init position
    _playerInitPos = doomGame.getGameState<GameState::PlayerPos>();
    _playerDistanceThreshold = _rnd()%768;

    // Reset models
    _model->reset();
    _agentModel->reset();
    if (_encoderModel != nullptr)
        _encoderModel->reset();

    return newMapId;
}

void Trainer::evaluateModel()
{
    TensorVector inputs(1);
    TensorVector outputs(3);

    double performanceAvg = 0.0;
    double performanceMin = std::numeric_limits<double>::max();
    int nSamples = 0;
    for (int i=0; i<doot2::batchSize; ++i) {
        auto storage = _sequenceStorage.read();
        auto* storageFrames = storage->getSequence<float>("frame");
        assert(storageFrames != nullptr);
        torch::Tensor pixelDataIn = storageFrames->tensor();
        for (int t=0; t<storageFrames->length(); ++t) {
            using namespace torch::indexing;
            inputs[0] = pixelDataIn.index({t, i, "..."}).unsqueeze(0).permute({0, 3, 1, 2});
            _model->infer(inputs, outputs);

            // Images output
            _trainingInfo.images["evaluation_input"].write()->copyFrom(
                inputs[0].permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo.images["evaluation_output"].write()->copyFrom(
                outputs[0].permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            torch::Tensor encodingImage;
            encodingImage = outputs[1].to(torch::kCPU, torch::kFloat32).reshape({32, 64})*0.05 + 0.5;
            encodingImage = tf::interpolate(encodingImage.unsqueeze(0).unsqueeze(0),
                tf::InterpolateFuncOptions()
                    .size(std::vector<long>{32*8, 64*8})
                    .mode(torch::kNearestExact).align_corners(false)
            );
            _trainingInfo.images["evaluation_encoding"].write()->copyFrom(encodingImage.contiguous().data_ptr<float>());
            torch::Tensor encodingMaskImage;
            encodingMaskImage = outputs[2].to(torch::kCPU, torch::kFloat32).reshape({32, 64});
            encodingMaskImage = tf::interpolate(encodingMaskImage.unsqueeze(0).unsqueeze(0),
                tf::InterpolateFuncOptions()
                    .size(std::vector<long>{32*8, 64*8})
                    .mode(torch::kNearestExact).align_corners(false)
            );
            _trainingInfo.images["evaluation_encoding_mask"].write()->copyFrom(encodingMaskImage.contiguous().data_ptr<float>());

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

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

    // Are we running the game inline or loading training data from the cache?
    bool runningGame = _experimentConfig["training_task"].get<int32_t>() == 1 || // agent policy task
        recordingToCache();

    bool recording = false;
    bool evaluating = false;
    int recordedFrameId = 0;
    int epoch = 0;
    _quit = false;
    while (!_quit) {
        if (runningGame || evaluating) {
            // Map the frame into a tensor
            frameTV[0] = torch::from_blob(frameYUVData.data(),
                {
                    1 /*batch size*/, doot2::frameWidth, doot2::frameHeight,
                    getImageFormatNChannels(ImageFormat::YUV)
                },
                torch::TensorOptions().device(torch::kCPU)
            );

            // Run encoder and agent model inference
            if (_encoderModel == nullptr) {
                // no encoder model in use, input raw frames to the model
                _agentModel->infer(frameTV, actionTV);
            } else {
                _encoderModel->infer(frameTV, encodingTV);
                _agentModel->infer(encodingTV, actionTV);
            }

            // Convert agent model output to an action for this timestep
            auto action = _actionConverter(actionTV[0]);

            // Update the game state, restart if required
            if (recordedFrameId >= _sequenceStorage.length() || doomGame.update(action)) {
                int nextBatchEntryId = _batchEntryId + 1;
                if (recordedFrameId < _sequenceStorage.length())
                    nextBatchEntryId = _batchEntryId;
                nextMap(nextBatchEntryId, evaluating);
                recordedFrameId = 0;
                recording = false;
                continue;
            }

            // Copy the DoomGame frame to local storage
            {
                auto frameHandle = _frame.write();
                frameHandle->copyFrom(doomGame.getPixelsBGRA());
            }

            // Recording to cache finished, move saved sequence batch to frame cache
            if (recordingToCache() && _newPatchReady) {
                printf("Cache frame sequence %s record finished, new patch ready.\n",
                    _frameCacheRecordSequenceName.c_str());
                fs::rename(fs::path("temp") / _frameCacheRecordSequenceName,
                    _frameCachePath / _frameCacheRecordSequenceName);
                _frameCacheRecordSequenceName.clear();
                if (recordingToCache()) { // we're still missing sequences from the cache
                    _visitedMaps.clear();
                    _newPatchReady = false;
                }
                else { // all necessary sequences captured for the cache, proceed to training
                    runningGame = false;
                    continue;
                }
            }

            // Record
            if (recording) {
                if (recordingToCache()) {
                    recordFrameToCache(recordedFrameId);
                }
                else {
                    _sequenceStorage.getBatch<Action>("action", recordedFrameId)[_batchEntryId] = action;
                    {   // convert the frame
                        auto frameHandle = _frame.read();
                        convertImage(*frameHandle, frameYUV, ImageFormat::YUV);
                        _sequenceStorage.getBatch<float>("frame", recordedFrameId)[_batchEntryId] = torch::from_blob(
                            frameYUVData.data(), frameShape, torch::TensorOptions().device(torch::kCPU)
                        );
                    }
                    _sequenceStorage.getBatch<double>("reward",
                        recordedFrameId)[_batchEntryId] = 0.0; // TODO no rewards for now
                }
                ++recordedFrameId;
            } else if (startRecording()) {
                recording = true;
            }
        }
        else {
            printf("Loading sequence from frame cache, offset %d\n", epoch);
            loadSequenceStorageFromFrameCache(epoch);
            _newPatchReady = true;
        }

        // Train or evaluate
        if (_newPatchReady) {
            _model->waitForTrainingFinished();
            // quit() might've been called in the meanwhile
            if (_quit) break;

            if (evaluating) {
                saveExperiment(); // TODO save experiment to "latest", requires functionality for selecting the save dir
                evaluateModel();
                // TODO add experiment saving to "best" in case it outperforms the last best
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
        if (!experimentConfig.contains("use_frame_cache"))
            throw std::runtime_error("Experiment config does not contain entry \"use_frame_cache\"");
        if (experimentConfig["use_frame_cache"].get<bool>()) { // using cache, check for parameters
            if (!experimentConfig.contains("frame_cache_path"))
                throw std::runtime_error("Experiment config does not contain entry \"frame_cache_path\"");
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

    // Create frame cache path directory in case it's required and it doesn't already exist
    if (_experimentConfig["training_task"].get<int32_t>() == 0 && // Image encoding
        _experimentConfig["use_frame_cache"].get<bool>()) {
        _frameCachePath = _experimentConfig["frame_cache_path"].get<fs::path>();
        fs::create_directories(_frameCachePath);
    }

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
    TensorVector outputs(3);

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

int Trainer::nFrameCacheSequences()
{
    int n = 0;
    for (const auto& s : fs::directory_iterator(_frameCachePath))
        ++n;
    return n;
}

bool Trainer::recordingToCache()
{
    return (_experimentConfig["training_task"].get<int32_t>() == 0 && // frame encoding task
        _experimentConfig["use_frame_cache"].get<bool>() && // and using frame cache
        nFrameCacheSequences() < _experimentConfig["n_cached_sequences"].get<int>()); // but not sufficient amount of sequences in the frame cache
}

void Trainer::recordFrameToCache(int frameId)
{
    if (_frameCacheRecordSequenceName.empty()) {
        _frameCacheRecordSequenceName = [](){
            using namespace std::chrono;
            std::stringstream ss;
            auto now = system_clock::to_time_t(system_clock::now());
            ss << std::put_time(std::gmtime(&now), "%Y%m%dT%H%M%S");
            return ss.str();
        }();
    }
    std::stringstream batchEntryIdSs, frameIdSs;
    batchEntryIdSs << std::setw(5) << std::setfill('0') << _batchEntryId;
    frameIdSs << std::setw(5) << std::setfill('0') << frameId << ".png";

    fs::path frameFilename = fs::path("temp") / _frameCacheRecordSequenceName / batchEntryIdSs.str() / frameIdSs.str();

    fs::create_directories(frameFilename.parent_path());
    writeImageToFile(*_frame.read(), frameFilename);
}

void Trainer::loadSequenceStorageFromFrameCache(int offset)
{
    static std::vector<float> frameYUVData(doot2::frameWidth*doot2::frameHeight*
        getImageFormatNChannels(ImageFormat::YUV)); // external buffer to allow mapping to tensor
    static Image<float> frameYUV(doot2::frameWidth, doot2::frameHeight, ImageFormat::YUV, frameYUVData.data());
    static const std::vector<int64_t> frameShape{doot2::frameHeight, doot2::frameWidth,
        getImageFormatNChannels(ImageFormat::YUV)};

    auto frameCacheSequencePaths = [this](){
        std::vector<fs::path> paths;
        for (const auto& s : fs::directory_iterator(_frameCachePath)) {
            paths.push_back(_frameCachePath / s);
        }
        return paths;
    }();

    int nSequences = _experimentConfig["n_cached_sequences"].get<int>();
    if (nSequences > frameCacheSequencePaths.size())
        throw std::runtime_error("Number of sequences requested exceeds the n. of sequences cached\n");

    for (int i=0; i<doot2::sequenceLength; ++i) {
        auto& base = frameCacheSequencePaths[(offset + i) % nSequences];
        std::stringstream frameFilename;
        frameFilename << std::setw(5) << std::setfill('0') << i << ".png";
        for (int b=0; b<doot2::batchSize; ++b) {
            std::stringstream batchEntryDir;
            batchEntryDir << std::setw(5) << std::setfill('0') << b;
            fs::path filename = base / batchEntryDir.str() / frameFilename.str();
            //printf("loading from %s\n", filename.c_str());

            auto frame = readImageFromFile<uint8_t>(filename);
            convertImage(frame, frameYUV, ImageFormat::YUV);
            _sequenceStorage.getBatch<float>("frame", i)[b] = torch::from_blob(
                frameYUVData.data(), frameShape, torch::TensorOptions().device(torch::kCPU)
            );
        }
    }
}

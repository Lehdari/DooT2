//
// Project: DooT2
// File: MultiLevelAutoEncoderModel.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/models/MultiLevelAutoEncoderModel.hpp"
#include "ml/TrainingInfo.hpp"
#include "Constants.hpp"

#include <gvizdoom/DoomGame.hpp>

#include <filesystem>
#include <random>


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;
using namespace std::chrono;


namespace {

    INLINE torch::Tensor yuvLoss(const torch::Tensor& target, const torch::Tensor& pred) {
        return torch::mean(torch::abs(target-pred), {0, 2, 3}) // reduce only batch and spatial dimensions, preserve channels
            .dot(torch::tensor({1.8f, 1.1f, 1.1f}, // higher weight on Y channel
                TensorOptions().device(target.device())));
    }

    // Loss function for YUV image gradients
    inline torch::Tensor imageGradLoss(const torch::Tensor& target, const torch::Tensor& pred)
    {
        using namespace torch::indexing;

        // Pixel-space gradients
        torch::Tensor targetGradX =
            target.index({Slice(), Slice(), Slice(), Slice(1, None)}) -
                target.index({Slice(), Slice(), Slice(), Slice(None, -1)});
        torch::Tensor targetGradY =
            target.index({Slice(), Slice(), Slice(1, None), Slice()}) -
                target.index({Slice(), Slice(), Slice(None, -1), Slice()});
        torch::Tensor predGradX =
            pred.index({Slice(), Slice(), Slice(), Slice(1, None)}) -
                pred.index({Slice(), Slice(), Slice(), Slice(None, -1)});
        torch::Tensor predGradY =
            pred.index({Slice(), Slice(), Slice(1, None), Slice()}) -
                pred.index({Slice(), Slice(), Slice(None, -1), Slice()});

        return yuvLoss(targetGradX, predGradX) + yuvLoss(targetGradY, predGradY);
    }

    // Loss function for YUV image laplacian
    inline torch::Tensor imageLaplacianLoss(const torch::Tensor& target, const torch::Tensor& pred)
    {
        static float laplacianKernelData[9] = {
            -0.5,   -1.0,   -0.5,
            -1.0,   6.0,    -1.0,
            -0.5,   -1.0,   -0.5
        };
        static torch::Tensor laplacianKernel = torch::from_blob(laplacianKernelData, {3,3})
            .repeat({3,1,1,1}).to(target.device());
        torch::Tensor targetLaplacian = tf::conv2d(target, laplacianKernel,
            tf::Conv2dFuncOptions().padding(1).groups(3));
        torch::Tensor predLaplacian = tf::conv2d(pred, laplacianKernel,
            tf::Conv2dFuncOptions().padding(1).groups(3));

        return yuvLoss(targetLaplacian, predLaplacian);
    }

} // namespace

nlohmann::json MultiLevelAutoEncoderModel::getDefaultModelConfig()
{
    nlohmann::json modelConfig;

    modelConfig["frame_encoder_filename"] = "frame_encoder.pt";
    modelConfig["frame_decoder_filename"] = "frame_decoder.pt";
    modelConfig["discriminator_filename"] = "discriminator.pt";
    modelConfig["optimizer_learning_rate"] = 0.001;
    modelConfig["optimizer_beta1"] = 0.9;
    modelConfig["optimizer_beta2"] = 0.999;
    modelConfig["optimizer_epsilon"] = 1.0e-8;
    modelConfig["optimizer_weight_decay"] = 0.0001;
    modelConfig["n_training_cycles"] = 16;
    modelConfig["virtual_batch_size"] = 32;
    modelConfig["frame_loss_weight"] = 1.5;
    modelConfig["frame_grad_loss_weight"] = 1.8;
    modelConfig["frame_laplacian_loss_weight"] = 1.5;
    modelConfig["use_encoding_mean_loss"] = true;
    modelConfig["encoding_mean_loss_weight"] = 0.001;
    modelConfig["use_encoding_codistance_loss"] = true;
    modelConfig["encoding_codistance_loss_weight"] = 0.004;
    modelConfig["use_covariance_loss"] = false;
    modelConfig["covariance_loss_weight"] = 0.0001;
    modelConfig["use_encoding_prev_distance_loss"] = true;
    modelConfig["encoding_prev_distance_loss_weight"] = 0.1;
    modelConfig["initial_loss_level"] = 0.0;
    modelConfig["target_loss"] = 0.2;

    return modelConfig;
}

MultiLevelAutoEncoderModel::MultiLevelAutoEncoderModel() :
    _optimizer                      (std::make_unique<torch::optim::AdamW>(
                                     std::vector<optim::OptimizerParamGroup>
                                     {_frameEncoder->parameters(), _frameDecoder->parameters()})),
    _discriminatorOptimizer         (std::make_unique<torch::optim::AdamW>(
                                     std::vector<optim::OptimizerParamGroup>
                                     {_discriminator->parameters()})),
    _device                         (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    _optimizerLearningRate          (0.001),
    _optimizerBeta1                 (0.9),
    _optimizerBeta2                 (0.999),
    _optimizerEpsilon               (1.0e-8),
    _optimizerWeightDecay           (0.0001),
    _nTrainingCycles                (16),
    _virtualBatchSize               (32),
    _frameLossWeight                (1.5),
    _frameGradLossWeight            (1.8),
    _frameLaplacianLossWeight       (1.5),
    _useEncodingMeanLoss            (true),
    _encodingMeanLossWeight         (0.001),
    _useEncodingCodistanceLoss      (true),
    _encodingCodistanceLossWeight   (0.004),
    _useCovarianceLoss              (false),
    _covarianceLossWeight           (0.0001),
    _useEncodingPrevDistanceLoss    (true),
    _encodingPrevDistanceLossWeight (0.1),
    _targetLoss                     (0.2),
    _lossLevel                      (0.0),
    _batchPixelDiff                 (1.0)
{
}

void MultiLevelAutoEncoderModel::init(const nlohmann::json& experimentConfig)
{
    auto& modelConfig = experimentConfig["model_config"];
    fs::path experimentRoot = doot2::experimentsDirectory / experimentConfig["experiment_root"].get<fs::path>();

    // Load torch model file names from the model config
    _frameEncoderFilename = experimentRoot / "frame_encoder.pt";
    if (modelConfig.contains("frame_encoder_filename"))
        _frameEncoderFilename = experimentRoot / modelConfig["frame_encoder_filename"].get<fs::path>();

    _frameDecoderFilename = experimentRoot / "frame_decoder.pt";
    if (modelConfig.contains("frame_decoder_filename"))
        _frameDecoderFilename = experimentRoot / modelConfig["frame_decoder_filename"].get<fs::path>();

    _discriminatorFilename = experimentRoot / "discriminator.pt";
    if (modelConfig.contains("discriminator_filename"))
        _discriminatorFilename = experimentRoot / modelConfig["discriminator_filename"].get<fs::path>();

    // Separate loading paths in case a base experiment is specified
    fs::path frameEncoderFilename = _frameEncoderFilename;
    fs::path frameDecoderFilename = _frameDecoderFilename;
    fs::path discriminatorFilename = _discriminatorFilename;
    if (experimentConfig.contains("experiment_base_root") && experimentConfig.contains("base_model_config")) {
        frameEncoderFilename = experimentConfig["experiment_base_root"].get<fs::path>() /
            experimentConfig["base_model_config"]["frame_encoder_filename"].get<fs::path>();
        frameDecoderFilename = experimentConfig["experiment_base_root"].get<fs::path>() /
            experimentConfig["base_model_config"]["frame_decoder_filename"].get<fs::path>();
        discriminatorFilename = experimentConfig["experiment_base_root"].get<fs::path>() /
            experimentConfig["base_model_config"]["discriminator_filename"].get<fs::path>();
    }

    // Load frame encoder
    if (fs::exists(frameEncoderFilename)) {
        printf("Loading frame encoder model from %s\n", frameEncoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameEncoderFilename);
        _frameEncoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame encoder model.\n", frameEncoderFilename.c_str()); // TODO logging
        *_frameEncoder = MultiLevelFrameEncoderImpl();
    }

    // Load frame decoder
    if (fs::exists(frameDecoderFilename)) {
        printf("Loading frame decoder model from %s\n", frameDecoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameDecoderFilename);
        _frameDecoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame decoder model.\n", frameDecoderFilename.c_str()); // TODO logging
        *_frameDecoder = MultiLevelFrameDecoderImpl();
    }

    // Load discriminator
    if (fs::exists(discriminatorFilename)) {
        printf("Loading frame decoder model from %s\n", discriminatorFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(discriminatorFilename);
        _discriminator->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new discriminator model.\n", discriminatorFilename.c_str()); // TODO logging
        *_discriminator = DiscriminatorImpl();
    }

    // Move model parameters to GPU if it's used
    _frameEncoder->to(_device);
    _frameDecoder->to(_device);
    _discriminator->to(_device);

    // Setup hyperparameters
    _optimizerLearningRate = modelConfig["optimizer_learning_rate"];
    _optimizerBeta1 = modelConfig["optimizer_beta1"];
    _optimizerBeta2 = modelConfig["optimizer_beta2"];
    _optimizerEpsilon = modelConfig["optimizer_epsilon"];
    _optimizerWeightDecay = modelConfig["optimizer_weight_decay"];
    _nTrainingCycles = modelConfig["n_training_cycles"];
    _virtualBatchSize = modelConfig["virtual_batch_size"];
    _frameLossWeight = modelConfig["frame_loss_weight"];
    _frameGradLossWeight = modelConfig["frame_grad_loss_weight"];
    _frameLaplacianLossWeight = modelConfig["frame_laplacian_loss_weight"];
    _useEncodingMeanLoss = modelConfig["use_encoding_mean_loss"];
    _encodingMeanLossWeight = modelConfig["encoding_mean_loss_weight"];
    _useEncodingCodistanceLoss = modelConfig["use_encoding_codistance_loss"];
    _encodingCodistanceLossWeight = modelConfig["encoding_codistance_loss_weight"];
    _useCovarianceLoss = modelConfig["use_covariance_loss"];
    _covarianceLossWeight = modelConfig["covariance_loss_weight"];
    _useEncodingPrevDistanceLoss = modelConfig["use_encoding_prev_distance_loss"];
    _encodingPrevDistanceLossWeight = modelConfig["encoding_prev_distance_loss_weight"];
    _targetLoss = modelConfig["target_loss"];
    _lossLevel = modelConfig["initial_loss_level"];

    // Setup optimizers
    _optimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_frameEncoder->parameters(), _frameDecoder->parameters()});
    dynamic_cast<torch::optim::AdamWOptions&>(_optimizer->param_groups()[0].options())
        .lr(_optimizerLearningRate)
        .betas({_optimizerBeta1, _optimizerBeta2})
        .eps(_optimizerEpsilon)
        .weight_decay(_optimizerWeightDecay);

    _discriminatorOptimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_discriminator->parameters()});
    dynamic_cast<torch::optim::AdamWOptions&>(_optimizer->param_groups()[0].options())
        .lr(_optimizerLearningRate)
        .betas({_optimizerBeta1, _optimizerBeta2})
        .eps(_optimizerEpsilon)
        .weight_decay(_optimizerWeightDecay);

    _trainingStartTime = high_resolution_clock::now();
}

void MultiLevelAutoEncoderModel::setTrainingInfo(TrainingInfo* trainingInfo)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    _trainingInfo = trainingInfo;
    assert(_trainingInfo != nullptr);

    {   // Initialize the time series
        auto timeSeriesWriteHandle = _trainingInfo->trainingTimeSeries.write();
        timeSeriesWriteHandle->addSeries<double>("time", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameGradLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLaplacianLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCodistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingMeanLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("covarianceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingPrevDistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminationLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminatorLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("loss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevel", 0.0);
        timeSeriesWriteHandle->addSeries<double>("maxEncodingCodistance", 0.0);
        timeSeriesWriteHandle->addSeries<double>("maxEncodingCovariance", 0.0);
    }
    // Initialize images
    {
        auto width = doomGame.getScreenWidth();
        auto height = doomGame.getScreenHeight();
        *(_trainingInfo->images)["input5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["codistance_matrix"].write() = Image<float>(
            doot2::batchSize*16, doot2::batchSize*16, ImageFormat::GRAY);
        if (_useCovarianceLoss) {
            *(_trainingInfo->images)["covariance_matrix"].write() = Image<float>(
                doot2::encodingLength, doot2::encodingLength, ImageFormat::GRAY);
        }
    }
}

void MultiLevelAutoEncoderModel::save()
{
    try {
        {
            printf("Saving frame encoder model to %s\n", _frameEncoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameEncoder->save(outputArchive);
            outputArchive.save_to(_frameEncoderFilename);
        }
        {
            printf("Saving frame decoder model to %s\n", _frameDecoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameDecoder->save(outputArchive);
            outputArchive.save_to(_frameDecoderFilename);
        }
        {
            printf("Saving discriminator model to %s\n", _discriminatorFilename.c_str());
            serialize::OutputArchive outputArchive;
            _discriminator->save(outputArchive);
            outputArchive.save_to(_discriminatorFilename);
        }
    }
    catch (const std::exception& e) {
        printf("Could not save the models: '%s'\n", e.what());
    }
}

void MultiLevelAutoEncoderModel::infer(const TensorVector& input, TensorVector& output)
{
    _frameEncoder->train(false);
    _frameDecoder->train(false);

    torch::Tensor in5 = input[0].to(_device);
    torch::Tensor in4, in3, in2, in1;
    if (_lossLevel > 3.0)
        in4 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
    else
        in4 = torch::zeros({in5.sizes()[0], 3, 240, 320}, TensorOptions().device(_device));
    if (_lossLevel > 2.0)
        in3 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));
    else
        in3 = torch::zeros({in5.sizes()[0], 3, 120, 160}, TensorOptions().device(_device));
    if (_lossLevel > 1.0)
        in2 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{60, 80}).mode(kArea));
    else
        in2 = torch::zeros({in5.sizes()[0], 3, 60, 80}, TensorOptions().device(_device));
    if (_lossLevel > 0.0)
        in1 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{30, 40}).mode(kArea));
    else
        in1 = torch::zeros({in5.sizes()[0], 3, 30, 40}, TensorOptions().device(_device));
    torch::Tensor in0 = tf::interpolate(in5,
        tf::InterpolateFuncOptions().size(std::vector<long>{15, 20}).mode(kArea));

    // Frame encode
    torch::Tensor enc = _frameEncoder->forward(in5, in4, in3, in2, in1, in0, _lossLevel);

    // Frame decode
    auto [out0, out1, out2, out3, out4, out5] = _frameDecoder->forward(enc, _lossLevel);

    // Level weights
    float levelWeight0 = (float)std::clamp(1.0-_lossLevel, 0.0, 1.0);
    float levelWeight1 = (float)std::clamp(1.0-std::abs(_lossLevel-1.0), 0.0, 1.0);
    float levelWeight2 = (float)std::clamp(1.0-std::abs(_lossLevel-2.0), 0.0, 1.0);
    float levelWeight3 = (float)std::clamp(1.0-std::abs(_lossLevel-3.0), 0.0, 1.0);
    float levelWeight4 = (float)std::clamp(1.0-std::abs(_lossLevel-4.0), 0.0, 1.0);
    float levelWeight5 = (float)std::clamp(_lossLevel-4.0, 0.0, 1.0);

    // Resize outputs
    out0 = tf::interpolate(out0, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out1 = tf::interpolate(out1, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out2 = tf::interpolate(out2, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out3 = tf::interpolate(out3, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out4 = tf::interpolate(out4, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out5 = tf::interpolate(out5, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));

    output.clear();
    output.push_back((
        levelWeight0 * out0 +
        levelWeight1 * out1 +
        levelWeight2 * out2 +
        levelWeight3 * out3 +
        levelWeight4 * out4 +
        levelWeight5 * out5)
        .to(input[0].device()));
}

void MultiLevelAutoEncoderModel::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // This model is used solely for training, trainingInfo must never be nullptr
    assert(_trainingInfo != nullptr);

    const auto sequenceLength = storage.length();

    // Load the whole storage's pixel data to the GPU
    auto* storageFrames = storage.getSequence<float>("frame");
    assert(storageFrames != nullptr);
    torch::Tensor pixelDataIn = storageFrames->tensor().to(_device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3}); // permute into TBCHW

    // Random sample the pixel diff to get an approximation
    for (int i=0; i<8; ++i) {
        // Select two frames from different sequences on timestep t
        int t = rnd() % sequenceLength;
        int b1 = rnd() % doot2::batchSize;
        int b2 = rnd() % doot2::batchSize;
        while (b2 == b1)
            b2 = rnd() % doot2::batchSize;

        // Lowpass filter the diff to prevent sudden changes
        _batchPixelDiff = _batchPixelDiff*0.95 +
            0.05 * torch::mean(torch::abs(pixelDataIn.index({t, b1})-pixelDataIn.index({t, b2}))).item<double>();
    }

    // Set training mode on
    _frameEncoder->train(true);
    _frameDecoder->train(true);
    _discriminator->train(true);

    // Zero out the gradients
    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();
    _discriminator->zero_grad();

    // Used as a target for encoding distance matrix (try to get all encoding codistances to 1)
    torch::Tensor targetDistanceMatrix = torch::ones({doot2::batchSize, doot2::batchSize},
        TensorOptions().device(_device));
    torch::Tensor zero = torch::zeros({}, TensorOptions().device(_device));

    double lossAcc = 0.0; // accumulate loss over the optimization interval to compute average
    double frameLossAcc = 0.0;
    double frameGradLossAcc = 0.0;
    double frameLaplacianLossAcc = 0.0;
    double encodingCodistanceLossAcc = 0.0;
    double encodingMeanLossAcc = 0.0;
    double covarianceLossAcc = 0.0;
    double encodingPrevDistanceLossAcc = 0.0;
    double discriminationLossAcc = 0.0;
    double discriminatorLossAcc = 0.0;
    double maxEncodingCodistance = 1.0; // max distance between encodings
    double maxEncodingCovariance = 1.0; // max value in the covariance matrix
    int64_t nVirtualBatchesPerCycle = (int64_t)sequenceLength / _virtualBatchSize;
    const double framesPerCycle = (double)nVirtualBatchesPerCycle * _virtualBatchSize;
    for (int64_t c=0; c<_nTrainingCycles; ++c) {
        for (int64_t v=0; v<nVirtualBatchesPerCycle; ++v) {
            torch::Tensor in5, in4, in3, in2, in1, in0;
            torch::Tensor rOut0, rOut1, rOut2, rOut3, rOut4, rOut5;
            torch::Tensor covarianceMatrix, encPrev;
            _discriminator->train(true);
            for (int64_t b=0; b<_virtualBatchSize; ++b) {
                // frame (time point) to use this iteration
                int64_t t = v*_virtualBatchSize + b;

                // Prepare the inputs (original and scaled)
                in5 = pixelDataIn.index({(int)t});
                if (_lossLevel > 3.0)
                    in4 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
                else
                    in4 = torch::zeros({in5.sizes()[0], 3, 240, 320}, TensorOptions().device(_device));
                if (_lossLevel > 2.0)
                    in3 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));
                else
                    in3 = torch::zeros({in5.sizes()[0], 3, 120, 160}, TensorOptions().device(_device));
                if (_lossLevel > 1.0)
                    in2 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{60, 80}).mode(kArea));
                else
                    in2 = torch::zeros({in5.sizes()[0], 3, 60, 80}, TensorOptions().device(_device));
                if (_lossLevel > 0.0)
                    in1 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{30, 40}).mode(kArea));
                else
                    in1 = torch::zeros({in5.sizes()[0], 3, 30, 40}, TensorOptions().device(_device));
                in0 = tf::interpolate(in5, tf::InterpolateFuncOptions().size(std::vector<long>{15, 20}).mode(kArea));

                // Frame encode
                torch::Tensor enc = _frameEncoder(in5, in4, in3, in2, in1, in0, _lossLevel);

                // Compute distance from encoding mean to origin and encoding mean loss
                torch::Tensor encodingMeanLoss = torch::zeros({}, TensorOptions().device(_device));
                torch::Tensor encMean;
                if (_useEncodingMeanLoss or _useCovarianceLoss)
                    encMean = enc.mean(0);
                if (_useEncodingMeanLoss) {
                    encodingMeanLoss = encMean.norm() * _encodingMeanLossWeight;
                    encodingMeanLossAcc += encodingMeanLoss.item<double>();
                }

                // Compute distances between each encoding and the codistance loss
                torch::Tensor encodingCodistanceLoss = torch::zeros({}, TensorOptions().device(_device));
                torch::Tensor codistanceMatrix;
                if (_useEncodingCodistanceLoss) {
                    codistanceMatrix = torch::cdist(enc, enc);
                    maxEncodingCodistance = torch::max(codistanceMatrix).to(torch::kCPU).item<double>();
                    codistanceMatrix = codistanceMatrix +
                        torch::eye(doot2::batchSize, TensorOptions().device(_device));
                    encodingCodistanceLoss = torch::mse_loss(codistanceMatrix, targetDistanceMatrix) *
                        _encodingCodistanceLossWeight;
                    encodingCodistanceLossAcc += encodingCodistanceLoss.item<double>();
                }

                // Compute covariance matrix and covariance loss
                torch::Tensor covarianceLoss = torch::zeros({}, TensorOptions().device(_device));
                if (_useCovarianceLoss) {
                    torch::Tensor encDev = enc - encMean;
                    covarianceMatrix = ((encDev.transpose(0, 1).matmul(encDev)) /
                        (doot2::batchSize - 1.0f)).square();
                    maxEncodingCovariance = torch::max(covarianceMatrix).item<double>();
                    covarianceLoss = ((covarianceMatrix.sum() - covarianceMatrix.diagonal().sum()) /
                        (float) doot2::encodingLength) * _covarianceLossWeight;
                    covarianceLossAcc += covarianceLoss.item<double>();
                }

                // Loss from distance to previous encoding
                torch::Tensor encodingPrevDistanceLoss = torch::zeros({}, TensorOptions().device(_device));
                if (_useEncodingPrevDistanceLoss && b>0) {
                    // Use pixel difference to previous frames as a basis for target encoding distance to the previous
                    torch::Tensor prevPixelDiff = torch::mean(torch::abs(in5-pixelDataIn.index({(int)t-1})), {1, 2, 3});
                    // Normalize the difference using batchPixelDiff, the intuition being that encodings between
                    // sequences in the batch should have the distance of 1 from each other, as per mandated by the
                    // codistance loss.
                    torch::Tensor prevPixelDiffNormalized = prevPixelDiff / _batchPixelDiff;
                    torch::Tensor encPrevDistance = torch::linalg_norm(enc-encPrev, torch::nullopt, {1});
                    // Sometimes pixel diff between subsequent frames can be larger than batchPixelDiff, which will
                    // cause the normalized distance diff to be over 1. Scaling with 0.1 should give enough slack
                    // to prevent the individual sequence encoding domains from overlapping.
                    encodingPrevDistanceLoss = torch::mse_loss(encPrevDistance, prevPixelDiffNormalized*0.1f) *
                        _encodingPrevDistanceLossWeight *
                        ((float)_virtualBatchSize / ((float)_virtualBatchSize-1)); // normalize to match the actual v. batch size
                }
                encodingPrevDistanceLossAcc += encodingPrevDistanceLoss.item<double>();
                encPrev = enc.detach(); // stop gradient from flowing to the previous iteration

                // Frame decode
                auto [out0, out1, out2, out3, out4, out5] = _frameDecoder(enc, _lossLevel);

                // Frame loss weights
                float frameLossWeight0 = (float)std::clamp(1.0-_lossLevel, 0.0, 1.0);
                float frameLossWeight1 = (float)std::clamp(1.0-std::abs(_lossLevel-1.0), 0.0, 1.0);
                float frameLossWeight2 = (float)std::clamp(1.0-std::abs(_lossLevel-2.0), 0.0, 1.0);
                float frameLossWeight3 = (float)std::clamp(1.0-std::abs(_lossLevel-3.0), 0.0, 1.0);
                float frameLossWeight4 = (float)std::clamp(1.0-std::abs(_lossLevel-4.0), 0.0, 1.0);
                float frameLossWeight5 = (float)std::clamp(_lossLevel-4.0, 0.0, 1.0);

                // Frame decoding losses
                torch::Tensor frameLoss = _frameLossWeight*(
                    frameLossWeight0 * yuvLoss(in0, out0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * yuvLoss(in1, out1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * yuvLoss(in2, out2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * yuvLoss(in3, out3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * yuvLoss(in4, out4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * yuvLoss(in5, out5) : zero));
                frameLossAcc += frameLoss.item<double>();
                torch::Tensor frameGradLoss = _frameGradLossWeight*(
                    frameLossWeight0 * imageGradLoss(in0, out0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageGradLoss(in1, out1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageGradLoss(in2, out2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageGradLoss(in3, out3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageGradLoss(in4, out4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageGradLoss(in5, out5) : zero));
                frameGradLossAcc += frameGradLoss.item<double>();
                torch::Tensor frameLaplacianLoss = _frameLaplacianLossWeight*(
                    frameLossWeight0 * imageLaplacianLoss(in0, out0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageLaplacianLoss(in1, out1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageLaplacianLoss(in2, out2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageLaplacianLoss(in3, out3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageLaplacianLoss(in4, out4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageLaplacianLoss(in5, out5) : zero));
                frameLaplacianLossAcc += frameLaplacianLoss.item<double>();

                // Decoder discrimination loss
                torch::Tensor discriminationLoss;
                {
                    torch::Tensor randomEnc = createRandomEncodingInterpolations(enc);
                    std::tie(rOut0, rOut1, rOut2, rOut3, rOut4, rOut5) = _frameDecoder(randomEnc, _lossLevel);
                    constexpr double discriminationLossWeight = 0.1; // TODO make a hyperparameter
                    discriminationLoss = discriminationLossWeight * torch::l1_loss(torch::ones({doot2::batchSize},
                            TensorOptions().device(_device)),
                        _discriminator(rOut5, rOut4, rOut3, rOut2, rOut1, rOut0, _lossLevel));
                }
                discriminationLossAcc += discriminationLoss.item<double>();

                // Total loss
                torch::Tensor loss = encodingCodistanceLoss + encodingMeanLoss + covarianceLoss +
                    encodingPrevDistanceLoss + frameLoss + frameGradLoss + frameLaplacianLoss +
                    discriminationLoss;
                lossAcc += loss.item<double>();

                // Encoder-decoder backward pass
                loss.backward();

                // Discriminator training passes
                constexpr int discriminatorTrainingIterations = 2; // TODO make a hyperparameter
                for (int d=0; d<discriminatorTrainingIterations; ++d) {
                    _discriminator->zero_grad();
                    // Real (input) images
                    torch::Tensor discriminationReal = _discriminator(in5, in4, in3, in2, in1, in0, _lossLevel);
                    torch::Tensor discriminatorLoss = l1_loss(discriminationReal,
                        torch::ones_like(discriminationReal));
                    torch::Tensor discriminationFake;
                    // Fake (decoded) images
                    torch::Tensor randomEnc = createRandomEncodingInterpolations(
                        enc); // create new set of random interpolations
                    std::tie(rOut0, rOut1, rOut2, rOut3, rOut4, rOut5) = _frameDecoder(randomEnc, _lossLevel);
                    discriminationFake = _discriminator(
                        rOut5.detach(), rOut4.detach(), rOut3.detach(), rOut2.detach(), rOut1.detach(), rOut0.detach(),
                        _lossLevel);
                    discriminatorLoss += l1_loss(discriminationFake, torch::zeros_like(discriminationFake));
                    discriminatorLossAcc += discriminatorLoss.item<double>() / discriminatorTrainingIterations;
                    discriminatorLoss.backward();
                    _discriminatorOptimizer->step();
                }

                // Display
                if (b == 0) {
                    int displaySeqId = rnd() % doot2::batchSize;
                    torch::Tensor inImage0, inImage1, inImage2, inImage3, inImage4, inImage5;
                    torch::Tensor outImage0, outImage1, outImage2, outImage3, outImage4, outImage5;
                    torch::Tensor rOutImage0, rOutImage1, rOutImage2, rOutImage3, rOutImage4, rOutImage5;
                    scaleDisplayImages(
                        in0.index({displaySeqId}), in1.index({displaySeqId}), in2.index({displaySeqId}),
                        in3.index({displaySeqId}), in4.index({displaySeqId}), in5.index({displaySeqId}),
                        inImage0, inImage1, inImage2, inImage3, inImage4, inImage5, torch::kCPU
                    );
                    scaleDisplayImages(
                        out0.index({displaySeqId}), out1.index({displaySeqId}), out2.index({displaySeqId}),
                        out3.index({displaySeqId}), out4.index({displaySeqId}), out5.index({displaySeqId}),
                        outImage0, outImage1, outImage2, outImage3, outImage4, outImage5, torch::kCPU
                    );
                    scaleDisplayImages(
                        rOut0.index({displaySeqId}), rOut1.index({displaySeqId}), rOut2.index({displaySeqId}),
                        rOut3.index({displaySeqId}), rOut4.index({displaySeqId}), rOut5.index({displaySeqId}),
                        rOutImage0, rOutImage1, rOutImage2, rOutImage3, rOutImage4, rOutImage5, torch::kCPU
                    );

                    torch::Tensor codistanceMatrixCPU = codistanceMatrix.to(torch::kCPU);
                    codistanceMatrixCPU /= maxEncodingCodistance;
                    codistanceMatrixCPU = tf::interpolate(codistanceMatrixCPU.unsqueeze(0).unsqueeze(0),
                        tf::InterpolateFuncOptions().size(std::vector<long>{doot2::batchSize*16, doot2::batchSize*16})
                            .mode(kNearestExact).align_corners(false));
                    torch::Tensor covarianceMatrixCPU;
                    if (_useCovarianceLoss) {
                        covarianceMatrixCPU = covarianceMatrix.to(torch::kCPU);
                        covarianceMatrixCPU /= maxEncodingCovariance;
                        covarianceMatrixCPU = tf::interpolate(covarianceMatrixCPU.unsqueeze(0).unsqueeze(0),
                            tf::InterpolateFuncOptions().size(std::vector<long>{doot2::encodingLength, doot2::encodingLength})
                                .mode(kNearestExact).align_corners(false));
                    }

                    _trainingInfo->images["input5"].write()->copyFrom(inImage5.data_ptr<float>());
                    _trainingInfo->images["input4"].write()->copyFrom(inImage4.data_ptr<float>());
                    _trainingInfo->images["input3"].write()->copyFrom(inImage3.data_ptr<float>());
                    _trainingInfo->images["input2"].write()->copyFrom(inImage2.data_ptr<float>());
                    _trainingInfo->images["input1"].write()->copyFrom(inImage1.data_ptr<float>());
                    _trainingInfo->images["input0"].write()->copyFrom(inImage0.data_ptr<float>());
                    _trainingInfo->images["prediction5"].write()->copyFrom(outImage5.data_ptr<float>());
                    _trainingInfo->images["prediction4"].write()->copyFrom(outImage4.data_ptr<float>());
                    _trainingInfo->images["prediction3"].write()->copyFrom(outImage3.data_ptr<float>());
                    _trainingInfo->images["prediction2"].write()->copyFrom(outImage2.data_ptr<float>());
                    _trainingInfo->images["prediction1"].write()->copyFrom(outImage1.data_ptr<float>());
                    _trainingInfo->images["prediction0"].write()->copyFrom(outImage0.data_ptr<float>());
                    _trainingInfo->images["random5"].write()->copyFrom(rOutImage5.data_ptr<float>());
                    _trainingInfo->images["random4"].write()->copyFrom(rOutImage4.data_ptr<float>());
                    _trainingInfo->images["random3"].write()->copyFrom(rOutImage3.data_ptr<float>());
                    _trainingInfo->images["random2"].write()->copyFrom(rOutImage2.data_ptr<float>());
                    _trainingInfo->images["random1"].write()->copyFrom(rOutImage1.data_ptr<float>());
                    _trainingInfo->images["random0"].write()->copyFrom(rOutImage0.data_ptr<float>());
                    if (_useEncodingCodistanceLoss)
                        _trainingInfo->images["codistance_matrix"].write()->copyFrom(codistanceMatrixCPU.contiguous().data_ptr<float>());
                    if (_useCovarianceLoss)
                        _trainingInfo->images["covariance_matrix"].write()->copyFrom(covarianceMatrixCPU.contiguous().data_ptr<float>());
                }
            }

            // Apply gradients
            _optimizer->step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();
            _discriminator->zero_grad();

            if (_abortTraining)
                break;
        }

        lossAcc /= framesPerCycle;
        frameLossAcc /= framesPerCycle;
        frameGradLossAcc /= framesPerCycle;
        frameLaplacianLossAcc /= framesPerCycle;
        encodingCodistanceLossAcc /= framesPerCycle;
        encodingMeanLossAcc /= framesPerCycle;
        covarianceLossAcc /= framesPerCycle;
        encodingPrevDistanceLossAcc /= framesPerCycle;
        discriminationLossAcc /= framesPerCycle;
        discriminatorLossAcc /= framesPerCycle;

        // Loss level adjustment
        constexpr double controlP = 0.01;
        // use hyperbolic error metric (asymptotically larger adjustments when loss approaches 0)
        double error = 1.0 - (_targetLoss / (lossAcc + 1.0e-8));
        _lossLevel -= error*controlP; // P control should suffice
        _lossLevel = std::clamp(_lossLevel, 0.0, 5.0);

        // Write the time series
        {
            auto timeSeriesWriteHandle = _trainingInfo->trainingTimeSeries.write();
            auto currentTime = high_resolution_clock::now();

            timeSeriesWriteHandle->addEntries(
                "time", (double)duration_cast<milliseconds>(currentTime-_trainingStartTime).count() / 1000.0,
                "frameLoss", frameLossAcc,
                "frameGradLoss", frameGradLossAcc,
                "frameLaplacianLoss", frameLaplacianLossAcc,
                "encodingCodistanceLoss", encodingCodistanceLossAcc,
                "encodingMeanLoss", encodingMeanLossAcc,
                "covarianceLoss", covarianceLossAcc,
                "encodingPrevDistanceLoss", encodingPrevDistanceLossAcc,
                "discriminationLoss", discriminationLossAcc,
                "discriminatorLoss", discriminatorLossAcc,
                "loss", lossAcc,
                "lossLevel", _lossLevel,
                "maxEncodingCodistance", maxEncodingCodistance,
                "maxEncodingCovariance", maxEncodingCovariance
            );
        }

        lossAcc = 0.0;
        frameLossAcc = 0.0;
        frameGradLossAcc = 0.0;
        frameLaplacianLossAcc = 0.0;
        encodingCodistanceLossAcc = 0.0;
        encodingMeanLossAcc = 0.0;
        covarianceLossAcc = 0.0;
        encodingPrevDistanceLossAcc = 0.0;
        discriminationLossAcc = 0.0;
        discriminatorLossAcc = 0.0;

        if (_abortTraining)
            break;
    }
}

void MultiLevelAutoEncoderModel::scaleDisplayImages(
    const Tensor& orig0, const Tensor& orig1, const Tensor& orig2,
    const Tensor& orig3, const Tensor& orig4, const Tensor& orig5,
    Tensor& image0, Tensor& image1, Tensor& image2, Tensor& image3, Tensor& image4, Tensor& image5,
    torch::DeviceType device)
{
    image0 = tf::interpolate(orig0.unsqueeze(0), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image1 = tf::interpolate(orig1.unsqueeze(0), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image2 = tf::interpolate(orig2.unsqueeze(0), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image3 = tf::interpolate(orig3.unsqueeze(0), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image4 = tf::interpolate(orig4.unsqueeze(0), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image5 = orig5.permute({1, 2, 0}).contiguous().to(device);
}

torch::Tensor MultiLevelAutoEncoderModel::createRandomEncodingInterpolations(const Tensor& enc, double extrapolation)
{
    assert(extrapolation >= 0.0);

    // Squaring the random matrix creates more encodings closer to some of the original encodings.
    // Otherwise most of the interpolated entries would be close to an "average" encoding.
    torch::Tensor positiveBasis = tf::softmax(torch::pow(torch::randn({doot2::batchSize, doot2::batchSize},
        TensorOptions().device(enc.device())), 2.0f),
        tf::SoftmaxFuncOptions(1));
    // Domain extrapolation can be achieved by adding negative encoding contributions
    torch::Tensor negativeBasis = tf::softmax(torch::randn({doot2::batchSize, doot2::batchSize},
            TensorOptions().device(enc.device())),
        tf::SoftmaxFuncOptions(1));

    torch::Tensor basis = positiveBasis*(1.0+2.0*extrapolation) - negativeBasis*(2.0*extrapolation);
    return basis.matmul(enc);
}

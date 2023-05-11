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
    modelConfig["optimizer_learning_rate"] = 0.001;
    modelConfig["optimizer_beta1"] = 0.9;
    modelConfig["optimizer_beta2"] = 0.999;
    modelConfig["optimizer_epsilon"] = 1.0e-8;
    modelConfig["optimizer_weight_decay"] = 0.0001;
    modelConfig["training_iterations"] = 16*128;
    modelConfig["optimization_interval"] = 32;
    modelConfig["frame_loss_weight"] = 1.5;
    modelConfig["frame_grad_loss_weight"] = 1.8;
    modelConfig["frame_laplacian_loss_weight"] = 1.5;
    modelConfig["use_encoding_mean_loss"] = true;
    modelConfig["encoding_mean_loss_weight"] = 0.001;
    modelConfig["use_encoding_codistance_loss"] = true;
    modelConfig["encoding_codistance_loss_weight"] = 0.0005;
    modelConfig["use_covariance_loss"] = false;
    modelConfig["covariance_loss_weight"] = 0.0001;
    modelConfig["initial_loss_level"] = 0.0;
    modelConfig["target_loss"] = 0.2;

    return modelConfig;
}

MultiLevelAutoEncoderModel::MultiLevelAutoEncoderModel() :
    _optimizer                      (std::make_unique<torch::optim::AdamW>(
                                     std::vector<optim::OptimizerParamGroup>
                                     {_frameEncoder->parameters(), _frameDecoder->parameters()})),
    _device                         (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    _optimizerLearningRate          (0.001),
    _optimizerBeta1                 (0.9),
    _optimizerBeta2                 (0.999),
    _optimizerEpsilon               (1.0e-8),
    _optimizerWeightDecay           (0.0001),
    _nTrainingIterations            (16*128),
    _optimizationInterval           (32),
    _frameLossWeight                (1.5),
    _frameGradLossWeight            (1.8),
    _frameLaplacianLossWeight       (1.5),
    _useEncodingMeanLoss            (true),
    _encodingMeanLossWeight         (0.001),
    _useEncodingCodistanceLoss      (true),
    _encodingCodistanceLossWeight   (0.0005),
    _useCovarianceLoss              (false),
    _covarianceLossWeight           (0.0001),
    _targetLoss                     (0.2),
    _lossLevel                      (0.0)
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

    // Separate loading paths in case a base experiment is specified
    fs::path frameEncoderFilename = _frameEncoderFilename;
    fs::path frameDecoderFilename = _frameDecoderFilename;
    if (experimentConfig.contains("experiment_base_root") && experimentConfig.contains("base_model_config")) {
        frameEncoderFilename = experimentConfig["experiment_base_root"].get<fs::path>() /
            experimentConfig["base_model_config"]["frame_encoder_filename"].get<fs::path>();
        frameDecoderFilename = experimentConfig["experiment_base_root"].get<fs::path>() /
            experimentConfig["base_model_config"]["frame_decoder_filename"].get<fs::path>();
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

    // Move model parameters to GPU if it's used
    _frameEncoder->to(_device);
    _frameDecoder->to(_device);

    // Setup hyperparameters
    _optimizerLearningRate = modelConfig["optimizer_learning_rate"];
    _optimizerBeta1 = modelConfig["optimizer_beta1"];
    _optimizerBeta2 = modelConfig["optimizer_beta2"];
    _optimizerEpsilon = modelConfig["optimizer_epsilon"];
    _optimizerWeightDecay = modelConfig["optimizer_weight_decay"];
    _nTrainingIterations = modelConfig["training_iterations"];
    _optimizationInterval = modelConfig["optimization_interval"];
    _frameLossWeight = modelConfig["frame_loss_weight"];
    _frameGradLossWeight = modelConfig["frame_grad_loss_weight"];
    _frameLaplacianLossWeight = modelConfig["frame_laplacian_loss_weight"];
    _useEncodingMeanLoss = modelConfig["use_encoding_mean_loss"];
    _encodingMeanLossWeight = modelConfig["encoding_mean_loss_weight"];
    _useEncodingCodistanceLoss = modelConfig["use_encoding_codistance_loss"];
    _encodingCodistanceLossWeight = modelConfig["encoding_codistance_loss_weight"];
    _useCovarianceLoss = modelConfig["use_covariance_loss"];
    _covarianceLossWeight = modelConfig["covariance_loss_weight"];
    _targetLoss = modelConfig["target_loss"];
    _lossLevel = modelConfig["initial_loss_level"];

    // Setup optimizer
    _optimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_frameEncoder->parameters(), _frameDecoder->parameters()});
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
        auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
        timeSeriesWriteHandle->addSeries<double>("time", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameGradLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLaplacianLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCodistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingMeanLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("covarianceLoss", 0.0);
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

    // Load the whole storage's pixel data to the GPU
    auto* storageFrames = storage.getSequence<float>("frame");
    assert(storageFrames != nullptr);
    torch::Tensor pixelDataIn = storageFrames->tensor().to(_device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3});

    // Set training mode on
    _frameEncoder->train(true);
    _frameDecoder->train(true);

    // Zero out the gradients
    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();

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
    double maxEncodingCodistance = 1.0; // max distance between encodings
    double maxEncodingCovariance = 1.0; // max value in the covariance matrix
    const auto sequenceLength = storage.length();
    for (int64_t ti=0; ti<_nTrainingIterations; ++ti){
        // frame (time point) to use this iteration
        int64_t t = rnd() % sequenceLength;

        // Pick random sequence to display
        size_t displaySeqId = rnd() % storage.batchSize();

        // ID of the frame (in sequence) to be used in the training batch
        torch::Tensor in5 = pixelDataIn.index({(int)t});
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

        // Compute distance from encoding mean to origin and encoding mean loss
        torch::Tensor encodingMeanLoss, encMean;
        if (_useEncodingMeanLoss or _useCovarianceLoss)
            encMean = enc.mean(0);
        if (_useEncodingMeanLoss) {
            encodingMeanLoss = encMean.norm() * _encodingMeanLossWeight;
            encodingMeanLossAcc += encodingMeanLoss.item<double>();
        }

        // Compute distances between each encoding and the codistance loss
        torch::Tensor codistanceMatrix, encodingCodistanceLoss;
        if (_useEncodingCodistanceLoss) {
            codistanceMatrix = torch::cdist(enc, enc);
            maxEncodingCodistance = torch::max(codistanceMatrix).to(torch::kCPU).item<double>();
            codistanceMatrix = codistanceMatrix + maxEncodingCodistance *
                torch::eye(doot2::batchSize, TensorOptions().device(_device));
            encodingCodistanceLoss = torch::mse_loss(codistanceMatrix, targetDistanceMatrix) *
                _encodingCodistanceLossWeight;
            encodingCodistanceLossAcc += encodingCodistanceLoss.item<double>();
        }

        // Compute covariance matrix and covariance loss
        torch::Tensor covarianceMatrix, covarianceLoss;
        if (_useCovarianceLoss) {
            torch::Tensor encDev = enc - encMean;
            covarianceMatrix = ((encDev.transpose(0, 1).matmul(encDev)) /
                (doot2::batchSize - 1.0f)).square();
            maxEncodingCovariance = torch::max(covarianceMatrix).item<double>();
            covarianceLoss = ((covarianceMatrix.sum() - covarianceMatrix.diagonal().sum()) /
                (float) doot2::encodingLength) * _covarianceLossWeight;
            covarianceLossAcc += covarianceLoss.item<double>();
        }

        // Frame decode
        auto [out0, out1, out2, out3, out4, out5] = _frameDecoder->forward(enc, _lossLevel);

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

        torch::Tensor loss = frameLoss + frameGradLoss + frameLaplacianLoss + encodingCodistanceLoss +
            encodingMeanLoss;

        lossAcc += loss.item<double>();

        // Backward pass
        loss.backward();

        // Apply gradients
        if (ti % _optimizationInterval == _optimizationInterval-1) {
            _optimizer->step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();

            // Display
            torch::Tensor in5CPU = in5.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor in4CPU = in4.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            in4CPU = tf::interpolate(in4CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor in3CPU = in3.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            in3CPU = tf::interpolate(in3CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor in2CPU = in2.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            in2CPU = tf::interpolate(in2CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor in1CPU = in1.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            in1CPU = tf::interpolate(in1CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor in0CPU = in0.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            in0CPU = tf::interpolate(in0CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor out5CPU = out5.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor out4CPU = out4.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            out4CPU = tf::interpolate(out4CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor out3CPU = out3.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            out3CPU = tf::interpolate(out3CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor out2CPU = out2.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            out2CPU = tf::interpolate(out2CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor out1CPU = out1.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            out1CPU = tf::interpolate(out1CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
            torch::Tensor out0CPU = out0.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            out0CPU = tf::interpolate(out0CPU.unsqueeze(0), tf::InterpolateFuncOptions().size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
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

            _trainingInfo->images["input5"].write()->copyFrom(in5CPU.data_ptr<float>());
            _trainingInfo->images["input4"].write()->copyFrom(in4CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["input3"].write()->copyFrom(in3CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["input2"].write()->copyFrom(in2CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["input1"].write()->copyFrom(in1CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["input0"].write()->copyFrom(in0CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction5"].write()->copyFrom(out5CPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction4"].write()->copyFrom(out4CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction3"].write()->copyFrom(out3CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction2"].write()->copyFrom(out2CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction1"].write()->copyFrom(out1CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction0"].write()->copyFrom(out0CPU.permute({0, 2, 3, 1}).contiguous().data_ptr<float>());
            if (_useEncodingCodistanceLoss)
                _trainingInfo->images["codistance_matrix"].write()->copyFrom(codistanceMatrixCPU.contiguous().data_ptr<float>());
            if (_useCovarianceLoss)
                _trainingInfo->images["covariance_matrix"].write()->copyFrom(covarianceMatrixCPU.contiguous().data_ptr<float>());

            lossAcc /= (double)_optimizationInterval;
            frameLossAcc /= (double)_optimizationInterval;
            frameGradLossAcc /= (double)_optimizationInterval;
            frameLaplacianLossAcc /= (double)_optimizationInterval;
            encodingCodistanceLossAcc /= (double)_optimizationInterval;
            encodingMeanLossAcc /= (double)_optimizationInterval;
            covarianceLossAcc /= (double)_optimizationInterval;

            // Loss level adjustment
            constexpr double controlP = 0.01;
            // use hyperbolic error metric (asymptotically larger adjustments when loss approaches 0)
            double error = 1.0 - (_targetLoss / (lossAcc + 1.0e-8));
            _lossLevel -= error*controlP; // P control should suffice
            _lossLevel = std::clamp(_lossLevel, 0.0, 5.0);

            // Write the time series
            {
                auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
                auto currentTime = high_resolution_clock::now();

                timeSeriesWriteHandle->addEntries(
                    "time", (double)duration_cast<milliseconds>(currentTime-_trainingStartTime).count() / 1000.0,
                    "frameLoss", frameLossAcc,
                    "frameGradLoss", frameGradLossAcc,
                    "frameLaplacianLoss", frameLaplacianLossAcc,
                    "encodingCodistanceLoss", encodingCodistanceLossAcc,
                    "encodingMeanLoss", encodingMeanLossAcc,
                    "covarianceLoss", covarianceLossAcc,
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
        }

        if (_abortTraining)
            break;
    }
}

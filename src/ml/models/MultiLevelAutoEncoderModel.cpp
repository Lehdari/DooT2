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

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDACachingAllocator.h>
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
    modelConfig["n_training_cycles"] = 4;
    modelConfig["virtual_batch_size"] = 8;
    modelConfig["frame_loss_weight"] = 4.0;
    modelConfig["frame_grad_loss_weight"] = 1.5;
    modelConfig["frame_laplacian_loss_weight"] = 1.5;
    modelConfig["use_encoding_mean_loss"] = true;
    modelConfig["encoding_mean_loss_weight"] = 0.001;
    modelConfig["use_encoding_codistance_loss"] = true;
    modelConfig["encoding_codistance_loss_weight"] = 0.4;
    modelConfig["use_encoding_distance_loss"] = true;
    modelConfig["encoding_distance_loss_weight"] = 0.001;
    modelConfig["use_encoding_covariance_loss"] = false;
    modelConfig["encoding_covariance_loss_weight"] = 0.01;
    modelConfig["use_encoding_prev_distance_loss"] = true;
    modelConfig["encoding_prev_distance_loss_weight"] = 0.2;
    modelConfig["use_encoding_circular_loss"] = true;
    modelConfig["encoding_circular_loss_weight"] = 0.25;
    modelConfig["use_discriminator"] = true;
    modelConfig["discrimination_loss_weight"] = 0.4;
    modelConfig["discriminator_virtual_batch_size"] = 8;
    modelConfig["initial_loss_level"] = 0.0;
    modelConfig["target_reconstruction_loss"] = 0.3;

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
    _nTrainingCycles                (4),
    _virtualBatchSize               (8),
    _frameLossWeight                (3.0),
    _frameGradLossWeight            (1.5),
    _frameLaplacianLossWeight       (1.5),
    _useEncodingMeanLoss            (true),
    _encodingMeanLossWeight         (0.001),
    _useEncodingCodistanceLoss      (true),
    _encodingCodistanceLossWeight   (0.4),
    _useEncodingDistanceLoss        (true),
    _encodingDistanceLossWeight     (0.001),
    _useEncodingCovarianceLoss      (false),
    _encodingCovarianceLossWeight   (0.01),
    _useEncodingPrevDistanceLoss    (true),
    _encodingPrevDistanceLossWeight (0.2),
    _useEncodingCircularLoss        (true),
    _encodingCircularLossWeight     (0.25),
    _useDiscriminator               (true),
    _discriminationLossWeight       (0.4),
    _discriminatorVirtualBatchSize  (8),
    _targetReconstructionLoss       (0.3),
    _lossLevel                      (0.0),
    _batchPixelDiff                 (1.0),
    _batchEncDiff                   (sqrt(doot2::encodingLength)),
    _frameEncoder                   (4, true)
{
}

void MultiLevelAutoEncoderModel::init(const nlohmann::json& experimentConfig)
{
    auto& modelConfig = experimentConfig["model_config"];
    fs::path experimentRoot = doot2::experimentsDirectory / experimentConfig["experiment_root"].get<fs::path>();

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
    _useEncodingDistanceLoss = modelConfig["use_encoding_distance_loss"];
    _encodingDistanceLossWeight = modelConfig["encoding_distance_loss_weight"];
    _useEncodingCovarianceLoss = modelConfig["use_encoding_covariance_loss"];
    _encodingCovarianceLossWeight = modelConfig["encoding_covariance_loss_weight"];
    _useEncodingPrevDistanceLoss = modelConfig["use_encoding_prev_distance_loss"];
    _encodingPrevDistanceLossWeight = modelConfig["encoding_prev_distance_loss_weight"];
    _useEncodingCircularLoss = modelConfig["use_encoding_circular_loss"];
    _encodingCircularLossWeight = modelConfig["encoding_circular_loss_weight"];
    _useDiscriminator = modelConfig["use_discriminator"];
    _discriminationLossWeight = modelConfig["discrimination_loss_weight"];
    _discriminatorVirtualBatchSize = modelConfig["discriminator_virtual_batch_size"];
    _targetReconstructionLoss = modelConfig["target_reconstruction_loss"];
    _lossLevel = modelConfig["initial_loss_level"];

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
        *_frameEncoder = MultiLevelFrameEncoderImpl(4, true);
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
    if (_useDiscriminator) {
        if (fs::exists(discriminatorFilename)) {
            printf("Loading frame decoder model from %s\n", discriminatorFilename.c_str()); // TODO logging
            serialize::InputArchive inputArchive;
            inputArchive.load_from(discriminatorFilename);
            _discriminator->load(inputArchive);
        } else {
            printf("No %s found. Initializing new discriminator model.\n",
                discriminatorFilename.c_str()); // TODO logging
            *_discriminator = DiscriminatorImpl();
        }
    }

    // Move model parameters to GPU if it's used
    _frameEncoder->to(_device, torch::kBFloat16);
    _frameDecoder->to(_device, torch::kBFloat16);
    _discriminator->to(_device, torch::kBFloat16);

    // Setup optimizers
    _optimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_frameEncoder->parameters(), _frameDecoder->parameters()});
    dynamic_cast<torch::optim::AdamWOptions&>(_optimizer->param_groups()[0].options())
        .lr(_optimizerLearningRate)
        .betas({_optimizerBeta1, _optimizerBeta2})
        .eps(_optimizerEpsilon)
        .weight_decay(_optimizerWeightDecay);

    if (_useDiscriminator) {
        _discriminatorOptimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
            {_discriminator->parameters()});
        dynamic_cast<torch::optim::AdamWOptions&>(_optimizer->param_groups()[0].options())
            .lr(_optimizerLearningRate)
            .betas({_optimizerBeta1, _optimizerBeta2})
            .eps(_optimizerEpsilon)
            .weight_decay(_optimizerWeightDecay);
    }

    // TODO move to Model base class
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
        timeSeriesWriteHandle->addSeries<double>("encodingDistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingMeanLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCovarianceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingPrevDistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminationLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminatorLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCircularLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("reconstructionLosses", 0.0);
        timeSeriesWriteHandle->addSeries<double>("auxiliaryLosses", 0.0);
        timeSeriesWriteHandle->addSeries<double>("loss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevel", 0.0);
        timeSeriesWriteHandle->addSeries<double>("maxEncodingCodistance", 0.0);
        timeSeriesWriteHandle->addSeries<double>("maxEncodingCovariance", 0.0);
    }
    // Initialize images
    {
        auto width = doomGame.getScreenWidth();
        auto height = doomGame.getScreenHeight();
        *(_trainingInfo->images)["input7"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input6"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction7"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction6"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random7"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random6"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random5"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random4"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random3"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["random0"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["encoding"].write() = Image<float>(64*8, 32*8, ImageFormat::GRAY);
        if (_useEncodingCodistanceLoss) {
            *(_trainingInfo->images)["codistance_matrix"].write() = Image<float>(
                doot2::batchSize*16, doot2::batchSize*16, ImageFormat::GRAY);
        }
        if (_useEncodingCovarianceLoss) {
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

    MultiLevelImage in;
    in.img7 = input[0].to(_device);
    if (_lossLevel > 5.0)
        in.img6 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
    else
        in.img6 = torch::zeros({in.img7.sizes()[0], 3, 240, 320}, TensorOptions().device(_device));
    if (_lossLevel > 4.0)
        in.img5 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));
    else
        in.img5 = torch::zeros({in.img7.sizes()[0], 3, 120, 160}, TensorOptions().device(_device));
    if (_lossLevel > 3.0)
        in.img4 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{60, 80}).mode(kArea));
    else
        in.img4 = torch::zeros({in.img7.sizes()[0], 3, 60, 80}, TensorOptions().device(_device));
    if (_lossLevel > 2.0)
        in.img3 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{30, 40}).mode(kArea));
    else
        in.img3 = torch::zeros({in.img7.sizes()[0], 3, 30, 40}, TensorOptions().device(_device));
    if (_lossLevel > 1.0)
        in.img2 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{15, 20}).mode(kArea));
    else
        in.img2 = torch::zeros({in.img7.sizes()[0], 3, 15, 20}, TensorOptions().device(_device));
    if (_lossLevel > 0.0)
        in.img1 = tf::interpolate(in.img7, tf::InterpolateFuncOptions().size(std::vector<long>{15, 10}).mode(kArea));
    else
        in.img1 = torch::zeros({in.img7.sizes()[0], 3, 15, 10}, TensorOptions().device(_device));
    in.img0 = tf::interpolate(in.img7,
        tf::InterpolateFuncOptions().size(std::vector<long>{5, 5}).mode(kArea));
    in.level = _lossLevel;
//
//    in.img7 = in.img7.to(torch::kBFloat16);
//    in.img6 = in.img6.to(torch::kBFloat16);
//    in.img5 = in.img5.to(torch::kBFloat16);
//    in.img4 = in.img4.to(torch::kBFloat16);
//    in.img3 = in.img3.to(torch::kBFloat16);
//    in.img2 = in.img2.to(torch::kBFloat16);
//    in.img1 = in.img1.to(torch::kBFloat16);
//    in.img0 = in.img0.to(torch::kBFloat16);

    _frameEncoder->to(torch::kFloat32);
    _frameDecoder->to(torch::kFloat32);

    // Frame encode
    torch::Tensor enc = _frameEncoder->forward(in);

    // Frame decode
    auto out = _frameDecoder->forward(enc, _lossLevel);

    // Level weights
    float levelWeight0 = (float)std::clamp(1.0-_lossLevel, 0.0, 1.0);
    float levelWeight1 = (float)std::clamp(1.0-std::abs(_lossLevel-1.0), 0.0, 1.0);
    float levelWeight2 = (float)std::clamp(1.0-std::abs(_lossLevel-2.0), 0.0, 1.0);
    float levelWeight3 = (float)std::clamp(1.0-std::abs(_lossLevel-3.0), 0.0, 1.0);
    float levelWeight4 = (float)std::clamp(1.0-std::abs(_lossLevel-4.0), 0.0, 1.0);
    float levelWeight5 = (float)std::clamp(1.0-std::abs(_lossLevel-5.0), 0.0, 1.0);
    float levelWeight6 = (float)std::clamp(1.0-std::abs(_lossLevel-6.0), 0.0, 1.0);
    float levelWeight7 = (float)std::clamp(_lossLevel-6.0, 0.0, 1.0);

    // Resize outputs
    out.img0 = tf::interpolate(out.img0, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img1 = tf::interpolate(out.img1, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img2 = tf::interpolate(out.img2, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img3 = tf::interpolate(out.img3, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img4 = tf::interpolate(out.img4, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img5 = tf::interpolate(out.img5, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));
    out.img6 = tf::interpolate(out.img6, tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false));

    output.clear();
    output.push_back((
        levelWeight0 * out.img0 +
        levelWeight1 * out.img1 +
        levelWeight2 * out.img2 +
        levelWeight3 * out.img3 +
        levelWeight4 * out.img4 +
        levelWeight5 * out.img5 +
        levelWeight6 * out.img6 +
        levelWeight7 * out.img7)
        .to(input[0].device()));
}

void MultiLevelAutoEncoderModel::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // This model is used solely for training, trainingInfo must never be nullptr
    assert(_trainingInfo != nullptr);

    at::autocast::set_autocast_gpu_dtype(torch::kBFloat16);
    at::autocast::set_enabled(true);

    const int sequenceLength = (int)storage.length();

    // Load the whole storage's pixel data to the GPU
    auto* storageFrames = storage.getSequence<float>("frame");
    auto seq = scaleSequences(storageFrames, sequenceLength);

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
            0.05 * torch::mean(torch::abs(seq.img7.index({t, b1})-seq.img7.index({t, b2}))).item<double>();

        MultiLevelImage in {
            seq.img0.index({(int)t}),
            seq.img1.index({(int)t}),
            seq.img2.index({(int)t}),
            seq.img3.index({(int)t}),
            seq.img4.index({(int)t}),
            seq.img5.index({(int)t}),
            seq.img6.index({(int)t}),
            seq.img7.index({(int)t}),
            _lossLevel
        };
        torch::Tensor enc = _frameEncoder(in);
        _batchEncDiff = _batchEncDiff*0.95 + 0.05*torch::norm(enc.index({b1})-enc.index({b2})).item<double>();
    }

    // Set training mode on
    _frameEncoder->train(true);
    _frameDecoder->train(true);
    _discriminator->train(true);
    _frameEncoder->to(torch::kBFloat16);
    _frameDecoder->to(torch::kBFloat16);

    // Zero out the gradients
    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();
    _discriminator->zero_grad();

    // Used as a target for encoding distance matrix (try to get all encoding codistances to 1)
    torch::Tensor targetDistanceMatrix = torch::ones({doot2::batchSize, doot2::batchSize},
        TensorOptions().device(_device));
    torch::Tensor zero = torch::zeros({}, TensorOptions().device(_device));

    // Target covariance matrix
    torch::Tensor targetCovarianceMatrix;
    if (_useEncodingCovarianceLoss)
        targetCovarianceMatrix = torch::eye(doot2::encodingLength, TensorOptions().device(_device));

    double lossAcc = 0.0; // accumulate loss over the optimization interval to compute average
    double frameLossAcc = 0.0;
    double frameGradLossAcc = 0.0;
    double frameLaplacianLossAcc = 0.0;
    double encodingCodistanceLossAcc = 0.0;
    double encodingDistanceLossAcc = 0.0;
    double encodingMeanLossAcc = 0.0;
    double encodingCovarianceLossAcc = 0.0;
    double encodingPrevDistanceLossAcc = 0.0;
    double discriminationLossAcc = 0.0;
    double discriminatorLossAcc = 0.0;
    double encodingCircularLossAcc = 0.0;
    double reconstructionLossesAcc = 0.0;
    double auxiliaryLossesAcc = 0.0;
    double maxEncodingCodistance = 1.0; // max distance between encodings
    double maxEncodingCovariance = 1.0; // max value in the covariance matrix
    int64_t nVirtualBatchesPerCycle = (int64_t)sequenceLength / _virtualBatchSize;
    const double framesPerCycle = (double)nVirtualBatchesPerCycle * _virtualBatchSize;
    for (int64_t c=0; c<_nTrainingCycles; ++c) {
        for (int64_t v=0; v<nVirtualBatchesPerCycle; ++v) {
            MultiLevelImage rOut;
            torch::Tensor covarianceMatrix, enc, encPrev, encRandom;

            // Discriminator training passes
            if (_useDiscriminator) {
                for (int d = 0; d < _discriminatorVirtualBatchSize; ++d) {
                    int t = rnd() % sequenceLength;

                    torch::Tensor discriminatorLoss;
                    {   // Real (input) images
                        // Prepare the inputs (original and scaled)
                        MultiLevelImage in{
                            seq.img0.index({(int) t}),
                            seq.img1.index({(int) t}),
                            seq.img2.index({(int) t}),
                            seq.img3.index({(int) t}),
                            seq.img4.index({(int) t}),
                            seq.img5.index({(int) t}),
                            seq.img6.index({(int) t}),
                            seq.img7.index({(int) t}),
                            _lossLevel
                        };

                        torch::Tensor discriminationReal = _discriminator(in);
                        discriminatorLoss = l1_loss(discriminationReal,
                            torch::ones_like(discriminationReal));
                    }
                    {   // Fake (decoded) images
                        encRandom = torch::randn({doot2::batchSize, doot2::encodingLength},
                            TensorOptions().device(_device).dtype(torch::kBFloat16));
                        MultiLevelImage rOut2 = _frameDecoder(encRandom, _lossLevel);
                        MultiLevelImage rOutDetached{
                            rOut2.img0.detach(),
                            rOut2.img1.detach(),
                            rOut2.img2.detach(),
                            rOut2.img3.detach(),
                            rOut2.img4.detach(),
                            rOut2.img5.detach(),
                            rOut2.img6.detach(),
                            rOut2.img7.detach(),
                            _lossLevel
                        };
                        torch::Tensor discriminationFake = _discriminator(rOutDetached);
                        discriminatorLoss += l1_loss(discriminationFake, torch::zeros_like(discriminationFake));
                    }
                    discriminatorLossAcc += discriminatorLoss.item<double>();
                    discriminatorLoss.backward();
                }
                _frameDecoder->zero_grad();
            }

            for (int64_t b=0; b<_virtualBatchSize; ++b) {
                // frame (time point) to use this iteration
                int t = v*_virtualBatchSize + b;

                // Prepare the inputs (original and scaled)
                MultiLevelImage in {
                    seq.img0.index({(int)t}),
                    seq.img1.index({(int)t}),
                    seq.img2.index({(int)t}),
                    seq.img3.index({(int)t}),
                    seq.img4.index({(int)t}),
                    seq.img5.index({(int)t}),
                    seq.img6.index({(int)t}),
                    seq.img7.index({(int)t}),
                    _lossLevel
                };

                // Frame encode
                enc = _frameEncoder(in);

                // Compute distance from encoding mean to origin and encoding mean loss
                torch::Tensor encodingMeanLoss = zero;
                torch::Tensor encMean;
                if (_useEncodingMeanLoss or _useEncodingCovarianceLoss)
                    encMean = enc.mean(0);
                if (_useEncodingMeanLoss) {
                    encodingMeanLoss = encMean.square().sum() * _encodingMeanLossWeight;
                    encodingMeanLossAcc += encodingMeanLoss.item<double>();
                }

                // Compute distances between each encoding and the codistance loss
                torch::Tensor encodingCodistanceLoss = zero;
                torch::Tensor codistanceMatrix = torch::zeros({doot2::batchSize, doot2::batchSize},
                    TensorOptions().device(_device));
                if (_useEncodingCodistanceLoss) {
                    codistanceMatrix = torch::cdist(enc, enc);
                    maxEncodingCodistance = torch::max(codistanceMatrix).to(torch::kCPU, torch::kFloat32).item<double>();
                    codistanceMatrix = codistanceMatrix +
                        torch::eye(doot2::batchSize, TensorOptions().device(_device));
                    encodingCodistanceLoss = torch::mse_loss(codistanceMatrix, targetDistanceMatrix) *
                        _encodingCodistanceLossWeight;
                    encodingCodistanceLossAcc += encodingCodistanceLoss.item<double>();
                }

                // Compute mean encoding distance to origin loss
                torch::Tensor encodingDistanceLoss = zero;
                if (_useEncodingDistanceLoss) {
                    torch::Tensor squaredDistances = enc.square().sum({1});
                    encodingDistanceLoss = torch::mse_loss(squaredDistances.mean(),
                        torch::ones({}, TensorOptions().device(_device)) * (double)doot2::encodingLength) *
                        (_encodingDistanceLossWeight / doot2::encodingLength);
                    encodingDistanceLossAcc += encodingDistanceLoss.item<double>();
                }

                // Compute covariance matrix and covariance loss
                torch::Tensor encodingCovarianceLoss = zero;
                if (_useEncodingCovarianceLoss) {
                    torch::Tensor encDev = enc - encMean;
                    covarianceMatrix = (encDev.transpose(0, 1).matmul(encDev)) / (doot2::batchSize - 1.0f);
                    maxEncodingCovariance = torch::max(covarianceMatrix).item<double>();
                    torch::Tensor covarianceMatrixSquaredDiff = (covarianceMatrix - targetCovarianceMatrix).square();
                    encodingCovarianceLoss = _encodingCovarianceLossWeight * (covarianceMatrixSquaredDiff.mean() +
                        covarianceMatrixSquaredDiff.max().sqrt());
                    encodingCovarianceLossAcc += encodingCovarianceLoss.item<double>();
                }

                // Loss from distance to previous encoding
                torch::Tensor encodingPrevDistanceLoss = zero;
                if (_useEncodingPrevDistanceLoss && b>0) {
                    // Use pixel difference to previous frames as a basis for target encoding distance to the previous
                    torch::Tensor prevPixelDiff = (in.img7-seq.img7.index({(int)t-1})).abs().mean({1, 2, 3});
                    torch::Tensor prevPixelDiffNormalized = prevPixelDiff / _batchPixelDiff;
                    torch::Tensor encPrevDistanceNormalized = torch::linalg_norm(enc-encPrev, torch::nullopt, {1}) /
                        _batchEncDiff;
                    encodingPrevDistanceLoss = torch::mse_loss(encPrevDistanceNormalized, prevPixelDiffNormalized) *
                        _encodingPrevDistanceLossWeight *
                        ((float)_virtualBatchSize / ((float)_virtualBatchSize-1)); // normalize to match the actual v. batch size
                }
                encodingPrevDistanceLossAcc += encodingPrevDistanceLoss.item<double>();
                encPrev = enc.detach(); // stop gradient from flowing to the previous iteration

                // Frame decode
                auto out = _frameDecoder(enc, _lossLevel);

                // Frame loss weights
                float frameLossWeight0 = (float)std::clamp(1.0-_lossLevel, 0.0, 1.0);
                float frameLossWeight1 = (float)std::clamp(1.0-std::abs(_lossLevel-1.0), 0.0, 1.0);
                float frameLossWeight2 = (float)std::clamp(1.0-std::abs(_lossLevel-2.0), 0.0, 1.0);
                float frameLossWeight3 = (float)std::clamp(1.0-std::abs(_lossLevel-3.0), 0.0, 1.0);
                float frameLossWeight4 = (float)std::clamp(1.0-std::abs(_lossLevel-4.0), 0.0, 1.0);
                float frameLossWeight5 = (float)std::clamp(1.0-std::abs(_lossLevel-5.0), 0.0, 1.0);
                float frameLossWeight6 = (float)std::clamp(1.0-std::abs(_lossLevel-6.0), 0.0, 1.0);
                float frameLossWeight7 = (float)std::clamp(_lossLevel-6.0, 0.0, 1.0);

                if (_useDiscriminator || _useEncodingCircularLoss) {
                    encRandom = torch::randn({doot2::batchSize, doot2::encodingLength},
                        TensorOptions().device(_device).dtype(torch::kBFloat16));
                    rOut = _frameDecoder(encRandom, _lossLevel);
                }

                // Decoder discrimination loss
                torch::Tensor discriminationLoss = zero;
                if (_useDiscriminator) {
                    discriminationLoss = _discriminationLossWeight * torch::l1_loss(torch::ones({doot2::batchSize},
                        TensorOptions().device(_device)), _discriminator(rOut));
                    discriminationLossAcc += discriminationLoss.item<double>();
                }

                // Circular encoding loss (demand that random encodings match after decode-encode)
                torch::Tensor encodingCircularLoss = zero;
                if (_useEncodingCircularLoss) {
                    torch::Tensor encCircular = _frameEncoder(rOut);
                    encodingCircularLoss = _encodingCircularLossWeight*
                        torch::mean(torch::norm(encCircular-encRandom, 2.0, 1) / _batchEncDiff);
                }
                encodingCircularLossAcc += encodingCircularLoss.item<double>();

                // Total auxiliary losses
                torch::Tensor auxiliaryLosses = encodingCodistanceLoss + encodingDistanceLoss + encodingMeanLoss +
                    encodingCovarianceLoss + encodingPrevDistanceLoss + encodingCircularLoss + discriminationLoss;
                auxiliaryLossesAcc += auxiliaryLosses.item<double>();

                // Frame decoding losses
                torch::Tensor frameLoss = _frameLossWeight*(
                    frameLossWeight0 * yuvLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * yuvLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * yuvLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * yuvLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * yuvLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * yuvLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight6 * yuvLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight7 * yuvLoss(in.img7, out.img7) : zero));
                frameLossAcc += frameLoss.item<double>();
                torch::Tensor frameGradLoss = _frameGradLossWeight*(
                    frameLossWeight0 * imageGradLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageGradLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageGradLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageGradLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageGradLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageGradLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight6 * imageGradLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight7 * imageGradLoss(in.img7, out.img7) : zero));
                frameGradLossAcc += frameGradLoss.item<double>();
                torch::Tensor frameLaplacianLoss = _frameLaplacianLossWeight*(
                    frameLossWeight0 * imageLaplacianLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageLaplacianLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageLaplacianLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageLaplacianLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageLaplacianLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageLaplacianLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight6 * imageLaplacianLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight7 * imageLaplacianLoss(in.img7, out.img7) : zero));
                frameLaplacianLossAcc += frameLaplacianLoss.item<double>();

                // Frame reconstruction losses
                torch::Tensor reconstructionLosses = frameLoss + frameGradLoss + frameLaplacianLoss;
                reconstructionLossesAcc += reconstructionLosses.item<double>();

                // Total loss
                torch::Tensor loss = reconstructionLosses + auxiliaryLosses;
                lossAcc += loss.item<double>();

                // Encoder-decoder backward pass
                loss.backward();

                // Display
                if (b == 0) {
                    int displaySeqId = rnd() % doot2::batchSize;

                    // Input images
                    MultiLevelImage inImage;
                    scaleDisplayImages(
                        MultiLevelImage {
                            in.img0.index({displaySeqId}), in.img1.index({displaySeqId}),
                            in.img2.index({displaySeqId}), in.img3.index({displaySeqId}),
                            in.img4.index({displaySeqId}), in.img5.index({displaySeqId}),
                            in.img6.index({displaySeqId}), in.img7.index({displaySeqId}), 0.0
                        },
                        inImage, torch::kCPU
                    );
                    _trainingInfo->images["input7"].write()->copyFrom(inImage.img7.data_ptr<float>());
                    _trainingInfo->images["input6"].write()->copyFrom(inImage.img6.data_ptr<float>());
                    _trainingInfo->images["input5"].write()->copyFrom(inImage.img5.data_ptr<float>());
                    _trainingInfo->images["input4"].write()->copyFrom(inImage.img4.data_ptr<float>());
                    _trainingInfo->images["input3"].write()->copyFrom(inImage.img3.data_ptr<float>());
                    _trainingInfo->images["input2"].write()->copyFrom(inImage.img2.data_ptr<float>());
                    _trainingInfo->images["input1"].write()->copyFrom(inImage.img1.data_ptr<float>());
                    _trainingInfo->images["input0"].write()->copyFrom(inImage.img0.data_ptr<float>());

                    // Output images
                    MultiLevelImage outImage;
                    scaleDisplayImages(
                        MultiLevelImage {
                            out.img0.index({displaySeqId}), out.img1.index({displaySeqId}),
                            out.img2.index({displaySeqId}), out.img3.index({displaySeqId}),
                            out.img4.index({displaySeqId}), out.img5.index({displaySeqId}),
                            out.img6.index({displaySeqId}), out.img7.index({displaySeqId}), 0.0
                        },
                        outImage, torch::kCPU
                    );
                    _trainingInfo->images["prediction7"].write()->copyFrom(outImage.img7.data_ptr<float>());
                    _trainingInfo->images["prediction6"].write()->copyFrom(outImage.img6.data_ptr<float>());
                    _trainingInfo->images["prediction5"].write()->copyFrom(outImage.img5.data_ptr<float>());
                    _trainingInfo->images["prediction4"].write()->copyFrom(outImage.img4.data_ptr<float>());
                    _trainingInfo->images["prediction3"].write()->copyFrom(outImage.img3.data_ptr<float>());
                    _trainingInfo->images["prediction2"].write()->copyFrom(outImage.img2.data_ptr<float>());
                    _trainingInfo->images["prediction1"].write()->copyFrom(outImage.img1.data_ptr<float>());
                    _trainingInfo->images["prediction0"].write()->copyFrom(outImage.img0.data_ptr<float>());

                    // Output images decoded from random encodings
                    if (_useDiscriminator) {
                        MultiLevelImage rOutImage;
                        scaleDisplayImages(
                            MultiLevelImage {
                                rOut.img0.index({displaySeqId}), rOut.img1.index({displaySeqId}),
                                rOut.img2.index({displaySeqId}), rOut.img3.index({displaySeqId}),
                                rOut.img4.index({displaySeqId}), rOut.img5.index({displaySeqId}),
                                rOut.img6.index({displaySeqId}), rOut.img7.index({displaySeqId}), 0.0
                            },
                            rOutImage, torch::kCPU
                        );
                        _trainingInfo->images["random7"].write()->copyFrom(rOutImage.img7.data_ptr<float>());
                        _trainingInfo->images["random6"].write()->copyFrom(rOutImage.img6.data_ptr<float>());
                        _trainingInfo->images["random5"].write()->copyFrom(rOutImage.img5.data_ptr<float>());
                        _trainingInfo->images["random4"].write()->copyFrom(rOutImage.img4.data_ptr<float>());
                        _trainingInfo->images["random3"].write()->copyFrom(rOutImage.img3.data_ptr<float>());
                        _trainingInfo->images["random2"].write()->copyFrom(rOutImage.img2.data_ptr<float>());
                        _trainingInfo->images["random1"].write()->copyFrom(rOutImage.img1.data_ptr<float>());
                        _trainingInfo->images["random0"].write()->copyFrom(rOutImage.img0.data_ptr<float>());
                    }

                    torch::Tensor codistanceMatrixCPU;
                    if (_useEncodingCodistanceLoss) {
                        codistanceMatrixCPU = codistanceMatrix.to(torch::kCPU, torch::kFloat32);
                        codistanceMatrixCPU /= maxEncodingCodistance;
                        codistanceMatrixCPU = tf::interpolate(codistanceMatrixCPU.unsqueeze(0).unsqueeze(0),
                            tf::InterpolateFuncOptions()
                                .size(std::vector<long>{doot2::batchSize * 16, doot2::batchSize * 16})
                                .mode(kNearestExact).align_corners(false)
                        );
                    }
                    torch::Tensor covarianceMatrixCPU;
                    if (_useEncodingCovarianceLoss) {
                        covarianceMatrixCPU = covarianceMatrix.to(torch::kCPU, torch::kFloat32);
                        covarianceMatrixCPU /= maxEncodingCovariance;
                        covarianceMatrixCPU = tf::interpolate(covarianceMatrixCPU.unsqueeze(0).unsqueeze(0),
                            tf::InterpolateFuncOptions()
                                .size(std::vector<long>{doot2::encodingLength, doot2::encodingLength})
                                .mode(kNearestExact).align_corners(false)
                        );
                    }

                    if (_useEncodingCodistanceLoss)
                        _trainingInfo->images["codistance_matrix"].write()->copyFrom(codistanceMatrixCPU.contiguous().data_ptr<float>());
                    if (_useEncodingCovarianceLoss)
                        _trainingInfo->images["covariance_matrix"].write()->copyFrom(covarianceMatrixCPU.contiguous().data_ptr<float>());

                    torch::Tensor encodingImage;
                    encodingImage = enc.index({displaySeqId}).to(torch::kCPU, torch::kFloat32).reshape({32, 64}) + 0.5;
                    encodingImage = tf::interpolate(encodingImage.unsqueeze(0).unsqueeze(0),
                        tf::InterpolateFuncOptions()
                            .size(std::vector<long>{32*8, 64*8})
                            .mode(kNearestExact).align_corners(false)
                    );
                    _trainingInfo->images["encoding"].write()->copyFrom(encodingImage.contiguous().data_ptr<float>());
                }

                c10::cuda::CUDACachingAllocator::emptyCache();
            }

            // Apply gradients
            _optimizer->step();
            _discriminatorOptimizer->step();
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
        encodingDistanceLossAcc /= framesPerCycle;
        encodingMeanLossAcc /= framesPerCycle;
        encodingCovarianceLossAcc /= framesPerCycle;
        encodingPrevDistanceLossAcc /= framesPerCycle;
        discriminationLossAcc /= framesPerCycle;
        discriminatorLossAcc /= (double)nVirtualBatchesPerCycle*(double)_discriminatorVirtualBatchSize;
        encodingCircularLossAcc /= framesPerCycle;
        reconstructionLossesAcc /= framesPerCycle;
        auxiliaryLossesAcc /= framesPerCycle;

        // Loss level adjustment
        constexpr double controlP = 0.01;
        // use hyperbolic error metric (asymptotically larger adjustments when loss approaches 0)
        double error = 1.0 - (_targetReconstructionLoss / (reconstructionLossesAcc + 1.0e-8));
        _lossLevel -= error*controlP; // P control should suffice
        _lossLevel = std::clamp(_lossLevel, 0.0, 7.0);

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
                "encodingDistanceLoss", encodingDistanceLossAcc,
                "encodingMeanLoss", encodingMeanLossAcc,
                "encodingCovarianceLoss", encodingCovarianceLossAcc,
                "encodingPrevDistanceLoss", encodingPrevDistanceLossAcc,
                "discriminationLoss", discriminationLossAcc,
                "discriminatorLoss", discriminatorLossAcc,
                "encodingCircularLoss", encodingCircularLossAcc,
                "reconstructionLosses", reconstructionLossesAcc,
                "auxiliaryLosses", auxiliaryLossesAcc,
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
        encodingDistanceLossAcc = 0.0;
        encodingMeanLossAcc = 0.0;
        encodingCovarianceLossAcc = 0.0;
        encodingPrevDistanceLossAcc = 0.0;
        discriminationLossAcc = 0.0;
        discriminatorLossAcc = 0.0;
        encodingCircularLossAcc = 0.0;
        reconstructionLossesAcc = 0.0;
        auxiliaryLossesAcc = 0.0;

        if (_abortTraining)
            break;
    }

    at::autocast::clear_cache();
    at::autocast::set_enabled(false);
}

MultiLevelImage MultiLevelAutoEncoderModel::scaleSequences(
    const Sequence<float>* storageFrames, int sequenceLength)
{
    assert(storageFrames != nullptr);
    MultiLevelImage image;
    image.img7 = storageFrames->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
    assert(image.img7.sizes()[0] == sequenceLength);

    // Create scaled sequence data
    // Initialize to zeros
    image.img6 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 240, 320},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img5 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 120, 160},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img4 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 60, 80},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img3 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 30, 40},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img2 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 15, 20},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img1 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 15, 10},
        TensorOptions().device(_device).dtype(torch::kBFloat16));
    image.img0 = torch::zeros({sequenceLength, image.img7.sizes()[1], image.img7.sizes()[2], 5, 5},
        TensorOptions().device(_device).dtype(torch::kBFloat16));

    for (int t=0; t<sequenceLength; ++t) {
//        if (_lossLevel > 4.0) // these limits are intentionally 1 less than elsewhere because lossLevel may increase during training
            image.img6.index_put_({t}, tf::interpolate(image.img7.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{240, 320}).mode(kArea)));
//        if (_lossLevel > 3.0)
            image.img5.index_put_({t}, tf::interpolate(image.img6.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{120, 160}).mode(kArea)));
//        if (_lossLevel > 2.0)
            image.img4.index_put_({t}, tf::interpolate(image.img5.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{60, 80}).mode(kArea)));
//        if (_lossLevel > 1.0)
            image.img3.index_put_({t}, tf::interpolate(image.img4.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{30, 40}).mode(kArea)));
//        if (_lossLevel > 0.0)
            image.img2.index_put_({t}, tf::interpolate(image.img3.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{15, 20}).mode(kArea)));
        image.img1.index_put_({t}, tf::interpolate(image.img2.index({t}), tf::InterpolateFuncOptions()
            .size(std::vector<long>{15, 10}).mode(kArea)));
        image.img0.index_put_({t}, tf::interpolate(image.img1.index({t}), tf::InterpolateFuncOptions()
            .size(std::vector<long>{5, 5}).mode(kArea)));
    }

    image.level = _lossLevel;
    return image;
}

void MultiLevelAutoEncoderModel::scaleDisplayImages(
    const MultiLevelImage& orig, MultiLevelImage& image, torch::DeviceType device)
{
    image.img0 = tf::interpolate(orig.img0.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img1 = tf::interpolate(orig.img1.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img2 = tf::interpolate(orig.img2.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img3 = tf::interpolate(orig.img3.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img4 = tf::interpolate(orig.img4.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img5 = tf::interpolate(orig.img5.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img6 = tf::interpolate(orig.img6.unsqueeze(0).to(torch::kFloat32), tf::InterpolateFuncOptions()
        .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
        .permute({0, 2, 3, 1}).squeeze().contiguous().to(device);
    image.img7 = orig.img7.permute({1, 2, 0}).contiguous().to(device, torch::kFloat32);
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

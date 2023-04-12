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


static constexpr double     learningRate            = 1.0e-3; // TODO
static constexpr int64_t    nTrainingIterations     = 3*128;


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;
using namespace std::chrono;


namespace {

    INLINE torch::Tensor yuvLoss(const torch::Tensor& target, const torch::Tensor& pred) {
        return torch::mean(torch::abs(target-pred), {0, 2, 3}) // reduce only batch and spatial dimensions, preserve channels
            .dot(torch::tensor({2.0f, 1.0f, 1.0f}, // higher weight on Y channel
                TensorOptions().device(target.device())));
    }

    // Loss function for YUV images
    inline torch::Tensor imageLoss(const torch::Tensor& target, const torch::Tensor& pred, float gradWeight = 2.0f)
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

        return
            yuvLoss(target, pred) +
                gradWeight * yuvLoss(targetGradX, predGradX) +
                gradWeight * yuvLoss(targetGradY, predGradY);
    }

}

MultiLevelAutoEncoderModel::MultiLevelAutoEncoderModel(nlohmann::json* experimentConfig) :
    Model                   (experimentConfig),
    _optimizationInterval   (32),
    _optimizer              ({
        _frameEncoder->parameters(),
        _frameDecoder->parameters()},
        torch::optim::AdamWOptions(learningRate).betas({0.9, 0.999}).weight_decay(0.001)),
    _trainingStartTime      (high_resolution_clock::now()),
    _lossLevel              (0.0)
{
    auto& modelConfig = (*_experimentConfig)["model_config"];
    fs::path experimentRoot = doot2::experimentsDirectory / (*_experimentConfig)["experiment_root"].get<fs::path>();

    // Load torch model file names from the model config
    _frameEncoderFilename = experimentRoot / "frame_encoder.pt";
    if (modelConfig.contains("frame_encoder_filename"))
        _frameEncoderFilename = experimentRoot / modelConfig["frame_encoder_filename"].get<fs::path>();
    else
        modelConfig["frame_encoder_filename"] = "frame_encoder.pt";

    _frameDecoderFilename = experimentRoot / "frame_decoder.pt";
    if (modelConfig.contains("frame_decoder_filename"))
        _frameDecoderFilename = experimentRoot / modelConfig["frame_decoder_filename"].get<fs::path>();
    else
        modelConfig["frame_decoder_filename"] = "frame_decoder.pt";

    // Load frame encoder
    if (fs::exists(_frameEncoderFilename)) {
        printf("Loading frame encoder model from %s\n", _frameEncoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(_frameEncoderFilename);
        _frameEncoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame encoder model.\n", _frameEncoderFilename.c_str()); // TODO logging
    }

    // Load frame decoder
    if (fs::exists(_frameDecoderFilename)) {
        printf("Loading frame decoder model from %s\n", _frameDecoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(_frameDecoderFilename);
        _frameDecoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame decoder model.\n", _frameDecoderFilename.c_str()); // TODO logging
    }

    // Setup hyperparameters
    if (modelConfig.contains("optimization_interval"))
        _optimizationInterval = modelConfig["optimization_interval"];
    else
        modelConfig["optimization_interval"] = _optimizationInterval;
}

void MultiLevelAutoEncoderModel::setTrainingInfo(TrainingInfo* trainingInfo)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    _trainingInfo = trainingInfo;
    assert(_trainingInfo != nullptr);

    {   // Initialize the time series
        auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
        timeSeriesWriteHandle->addSeries<double>("time", 0.0);
        timeSeriesWriteHandle->addSeries<double>("loss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevel", 0.0);
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

}

void MultiLevelAutoEncoderModel::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // This model is used solely for training, trainingInfo must never be nullptr
    assert(_trainingInfo != nullptr);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);

    // Load the whole storage's pixel data to the GPU
    auto* storageFrames = storage.getSequence<float>("frame");
    assert(storageFrames != nullptr);
    torch::Tensor pixelDataIn = storageFrames->tensor().to(device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3});

    // Zero out the gradients
    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();

    const auto sequenceLength = storage.length();
    for (int64_t ti=0; ti<nTrainingIterations; ++ti){
        // frame (time point) to use this iteration
        int64_t t = rnd() % sequenceLength;

        // Pick random sequence to display
        size_t displaySeqId = rnd() % storage.batchSize();

        // ID of the frame (in sequence) to be used in the training batch
        torch::Tensor in5 = pixelDataIn.index({(int)t});
        torch::Tensor in4 = tf::interpolate(in5,
            tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
        torch::Tensor in3 = tf::interpolate(in5,
            tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));
        torch::Tensor in2 = tf::interpolate(in5,
            tf::InterpolateFuncOptions().size(std::vector<long>{60, 80}).mode(kArea));
        torch::Tensor in1 = tf::interpolate(in5,
            tf::InterpolateFuncOptions().size(std::vector<long>{30, 40}).mode(kArea));
        torch::Tensor in0 = tf::interpolate(in5,
            tf::InterpolateFuncOptions().size(std::vector<long>{15, 20}).mode(kArea));

        // Frame encode
        torch::Tensor enc = _frameEncoder->forward(in5);

        // Frame decode
        auto [out0, out1, out2, out3, out4, out5] = _frameDecoder->forward(enc);

        // Frame losses
        torch::Tensor frameLoss0 = imageLoss(in0, out0);
        torch::Tensor frameLoss1 = imageLoss(in1, out1);
        torch::Tensor frameLoss2 = imageLoss(in2, out2);
        torch::Tensor frameLoss3 = imageLoss(in3, out3);
        torch::Tensor frameLoss4 = imageLoss(in4, out4);
        torch::Tensor frameLoss5 = imageLoss(in5, out5);

        // Frame loss weights
        float frameLossWeight0 = (float)std::clamp(1.0-_lossLevel, 0.0, 1.0);
        float frameLossWeight1 = (float)std::clamp(1.0-std::abs(_lossLevel-1.0), 0.0, 1.0);
        float frameLossWeight2 = (float)std::clamp(1.0-std::abs(_lossLevel-2.0), 0.0, 1.0);
        float frameLossWeight3 = (float)std::clamp(1.0-std::abs(_lossLevel-3.0), 0.0, 1.0);
        float frameLossWeight4 = (float)std::clamp(1.0-std::abs(_lossLevel-4.0), 0.0, 1.0);
        float frameLossWeight5 = (float)std::clamp(_lossLevel-4.0, 0.0, 1.0);

        // Total loss
        torch::Tensor loss =
            frameLossWeight0 * frameLoss0 +
            frameLossWeight1 * frameLoss1 +
            frameLossWeight2 * frameLoss2 +
            frameLossWeight3 * frameLoss3 +
            frameLossWeight4 * frameLoss4 +
            frameLossWeight5 * frameLoss5;

        // Loss level adjustment
        constexpr double targetLoss = 0.1;
        constexpr double controlP = 0.01;
        // use hyperbolic error metric (asymptotically larger adjustments when loss approaches 0)
        double error = 1.0 - (targetLoss / (loss.item<double>() + 1.0e-8));
        _lossLevel -= error*controlP; // P control should suffice
        _lossLevel = std::clamp(_lossLevel, 0.0, 5.0);

        // Backward pass
        loss.backward();

        // Apply gradients
        if (ti % _optimizationInterval == _optimizationInterval-1) {
            _optimizer.step();
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

            // Write the time series
            {
                auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
                auto currentTime = high_resolution_clock::now();

                timeSeriesWriteHandle->addEntries(
                    "time", (double)duration_cast<milliseconds>(currentTime-_trainingStartTime).count() / 1000.0,
                    "loss", loss.item<double>(),
                    "lossLevel", _lossLevel
                );
            }
        }

        if (_abortTraining)
            break;
    }
}

//
// Project: DooT2
// File: AutoEncoderModel.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/models/AutoEncoderModel.hpp"
#include "ml/TrainingInfo.hpp"
#include "util/SequenceStorage.hpp"
#include "Constants.hpp"

#include <gvizdoom/DoomGame.hpp>

#include <filesystem>
#include <random>


static constexpr double     learningRate            = 1.0e-3; // TODO
static constexpr int64_t    nTrainingIterations     = 4*64;


using namespace ml;
using namespace torch;
namespace tf = torch::nn::functional;
namespace fs = std::filesystem;
using namespace std::chrono;


#define FLOW 1
#define DOUBLE_EDEC 1


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


AutoEncoderModel::AutoEncoderModel() :
    _optimizer          ({
        _frameEncoder->parameters(),
        _frameDecoder->parameters(),
        _flowDecoder->parameters()},
        torch::optim::AdamWOptions(learningRate).betas({0.9, 0.999}).weight_decay(0.001)),
    _trainingStartTime  (high_resolution_clock::now())
{
    using namespace doot2;

    // Load frame encoder
    if (fs::exists(frameEncoderFilename)) {
        printf("Loading frame encoder model from %s\n", frameEncoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameEncoderFilename);
        _frameEncoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new frame encoder model.\n", frameEncoderFilename.c_str()); // TODO logging
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
    }

    // Load flow decoder
    if (fs::exists(flowDecoderFilename)) {
        printf("Loading frame decoder model from %s\n", flowDecoderFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(flowDecoderFilename);
        _flowDecoder->load(inputArchive);
    }
    else {
        printf("No %s found. Initializing new flow decoder model.\n", flowDecoderFilename.c_str()); // TODO logging
    }
}

void AutoEncoderModel::setTrainingInfo(TrainingInfo* trainingInfo)
{
    auto& doomGame = gvizdoom::DoomGame::instance();

    _trainingInfo = trainingInfo;
    assert(_trainingInfo != nullptr);

    {   // Initialize the time series
        auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
        timeSeriesWriteHandle->addSeries<double>("time", 0.0);
        timeSeriesWriteHandle->addSeries<double>("loss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("flowForwardLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("flowBackwardLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameLossDoubleEdec", 0.0);
        timeSeriesWriteHandle->addSeries<double>("doubleEncodingLoss", 0.0);
    }
    // Initialize images
    {
        auto width = doomGame.getScreenWidth();
        auto height = doomGame.getScreenHeight();
        *(_trainingInfo->images)["input_1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["input_1_scaled_1"].write() = Image<float>(width/2, height/2, ImageFormat::YUV);
        *(_trainingInfo->images)["input_1_scaled_2"].write() = Image<float>(width/4, height/4, ImageFormat::YUV);
        *(_trainingInfo->images)["input_2"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction_1"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction_1_scaled_1"].write() = Image<float>(width/2, height/2, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction_1_scaled_2"].write() = Image<float>(width/4, height/4, ImageFormat::YUV);
        *(_trainingInfo->images)["prediction_1_double_edec"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["flow_forward"].write() = Image<float>(width, height, ImageFormat::BGRA);
        *(_trainingInfo->images)["flow_forward_mapped"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["flow_forward_diff"].write() = Image<float>(width, height, ImageFormat::GRAY);
        *(_trainingInfo->images)["flow_backward"].write() = Image<float>(width, height, ImageFormat::BGRA);
        *(_trainingInfo->images)["flow_backward_mapped"].write() = Image<float>(width, height, ImageFormat::YUV);
        *(_trainingInfo->images)["flow_backward_diff"].write() = Image<float>(width, height, ImageFormat::GRAY);
    }
}

void AutoEncoderModel::trainImpl(SequenceStorage& storage)
{
    using namespace torch::indexing;
    static std::default_random_engine rnd(1507715517);

    // This model is used solely for training, trainingInfo must never be nullptr
    assert(_trainingInfo != nullptr);

    // Check CUDA availability
    torch::Device device{torch::kCPU};
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU); // shorthand for above

    // Move model parameters to GPU
    _frameEncoder->to(device);
    _frameDecoder->to(device);
    _flowDecoder->to(device);

    // Load the whole storage's pixel data to the GPU
    auto* storageFrames = storage.getSequence<float>("frame");
    assert(storageFrames != nullptr);
    torch::Tensor pixelDataIn = storageFrames->tensor().to(device);
    pixelDataIn = pixelDataIn.permute({0, 1, 4, 2, 3});

    Tensor flowBase = tf::affine_grid(torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        TensorOptions().device(device)).broadcast_to({storage.batchSize(), 2, 3}),
        {storage.batchSize(), 2, 480, 640});

    // Required for latent space normalization
    torch::Tensor encodingZeros = torch::zeros({doot2::encodingLength}, TensorOptions().device(device));
    torch::Tensor encodingOnes = torch::ones({doot2::encodingLength}, TensorOptions().device(device));

    _frameEncoder->zero_grad();
    _frameDecoder->zero_grad();
    _flowDecoder->zero_grad();

    const auto sequenceLength = storage.length();
    const int64_t optimizationInterval = 8; // for this many frames gradients will be accumulated before update
    for (int64_t ti=0; ti<nTrainingIterations; ++ti) {
        // frame (time point) to use this iteration
        int64_t t = rnd() % (sequenceLength-1);

        // Pick random sequence to display
        size_t displaySeqId = rnd() % storage.batchSize();

        // ID of the frame (in sequence) to be used in the training batch
        torch::Tensor batchIn1 = pixelDataIn.index({(int)t});
        torch::Tensor batchIn2 = pixelDataIn.index({(int)t+1});
        torch::Tensor batchIn1Scaled1 = tf::interpolate(batchIn1,
            tf::InterpolateFuncOptions().size(std::vector<long>{240, 320}).mode(kArea));
        torch::Tensor batchIn1Scaled2 = tf::interpolate(batchIn1,
            tf::InterpolateFuncOptions().size(std::vector<long>{120, 160}).mode(kArea));

        // Forward passes
        torch::Tensor encoding1 = _frameEncoder->forward(batchIn1); // image t encode
        torch::Tensor encoding2 = _frameEncoder->forward(batchIn2); // image t+1 encode
#if FLOW
        // Flow from frame t to t+1
        torch::Tensor flowForward = _flowDecoder->forward(encoding2-encoding1);
        torch::Tensor flowFrameForward = tf::grid_sample(batchIn1, flowBase+flowForward,
            tf::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kBorder)
            .align_corners(true));

        // Flow from frame t+1 to t
        torch::Tensor flowBackward = _flowDecoder->forward(encoding1-encoding2);
        torch::Tensor flowFrameBackward = tf::grid_sample(batchIn2, flowBase+flowBackward,
            tf::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true));
#endif
        // Frame decode
        auto tupleOut = _frameDecoder->forward(encoding1);
        auto& batchOut = std::get<0>(tupleOut);
        auto& batchOutScaled1 = std::get<1>(tupleOut);
        auto& batchOutScaled2 = std::get<2>(tupleOut);

#if DOUBLE_EDEC
        // Double encode-decode
        torch::Tensor encoding1double = _frameEncoder->forward(batchOut);
        auto tupleOut2 = _frameDecoder->forward(encoding1double);
        auto& batchOut2 = std::get<0>(tupleOut2);
        auto& batchOut2Scaled1 = std::get<1>(tupleOut2);
        auto& batchOut2Scaled2 = std::get<2>(tupleOut2);
#endif

        // Frame losses
        torch::Tensor frameLoss =
            imageLoss(batchIn1, batchOut) +
            imageLoss(batchIn1Scaled1, batchOutScaled1) +
            imageLoss(batchIn1Scaled2, batchOutScaled2);
#if DOUBLE_EDEC
        torch::Tensor frameLossDoubleEdec =
            imageLoss(batchIn1, batchOut2) +
            imageLoss(batchIn1Scaled1, batchOut2Scaled1) +
            imageLoss(batchIn1Scaled2, batchOut2Scaled2);
#endif
#if FLOW
        // Flow loss
        torch::Tensor flowForwardDiff = torch::abs(
            batchIn2.index({Slice(), 0, "..."}) -
            flowFrameForward.index({Slice(), 0, "..."}));
        torch::Tensor flowBackwardDiff = torch::abs(
            batchIn1.index({Slice(), 0, "..."}) -
            flowFrameBackward.index({Slice(), 0, "..."}));
        torch::Tensor flowForwardLoss = 5.0f * torch::mean(flowForwardDiff);
        torch::Tensor flowBackwardLoss = 5.0f * torch::mean(flowBackwardDiff);
#endif
        // Encoding mean/variance loss
        torch::Tensor encodingMean = torch::mean(encoding1, {0}); // mean across the batch dimension
        torch::Tensor encodingVar = torch::var(encoding1, 0); // variance across the batch dimension
        torch::Tensor encodingMeanLoss = 1.0e-6f * torch::sum(torch::square(encodingMean - encodingZeros));
        torch::Tensor encodingVarLoss = 1.0e-6f * torch::sum(torch::square(encodingVar - encodingOnes));
#if DOUBLE_EDEC
        // Double encoding loss
        torch::Tensor encoding1detached = encoding1.detach(); // prevent gradient flowing back through encoding1 so that the double edec loss won't pull it towards the worse double encoding
        torch::Tensor doubleEncodingLoss = 1.0e-4f * torch::mse_loss(encoding1detached, encoding1double);
#endif

        // Total loss
        torch::Tensor loss =
            frameLoss
#if FLOW
            + flowForwardLoss
            + flowBackwardLoss
#endif
#if DOUBLE_EDEC
            + frameLossDoubleEdec
            + doubleEncodingLoss
#endif
            + encodingMeanLoss
            + encodingVarLoss;

        if (_abortTraining)
            return;

        // Backward pass
        loss.backward();

        // Apply gradients
        if (ti % optimizationInterval == optimizationInterval-1) {
            _optimizer.step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();
            _flowDecoder->zero_grad();

            // Display
            torch::Tensor batchIn1CPU = batchIn1.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor batchIn1Scaled1CPU = batchIn1Scaled1.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor batchIn1Scaled2CPU = batchIn1Scaled2.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor batchIn2CPU = batchIn2.index({(int)displaySeqId, "..."}).to(torch::kCPU);
#if FLOW
            torch::Tensor flowForwardCPU = flowForward.index({(int)displaySeqId, "..."}).to(torch::kCPU) * 2.0f + 0.5f;
            flowForwardCPU = torch::cat({flowForwardCPU, torch::zeros({flowForwardCPU.sizes()[0], flowForwardCPU.sizes()[1], 1l})}, 2);
            flowForwardCPU = torch::cat({flowForwardCPU, torch::ones({flowForwardCPU.sizes()[0], flowForwardCPU.sizes()[1], 1l})}, 2);
            torch::Tensor flowFrameForwardCPU = flowFrameForward.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor flowForwardDiffCPU = flowForwardDiff.index({(int)displaySeqId, "..."}).contiguous().to(torch::kCPU);
            torch::Tensor flowBackwardCPU = flowBackward.index({(int)displaySeqId, "..."}).to(torch::kCPU) * 2.0f + 0.5f;
            flowBackwardCPU = torch::cat({flowBackwardCPU, torch::zeros({flowBackwardCPU.sizes()[0], flowBackwardCPU.sizes()[1], 1l})}, 2);
            flowBackwardCPU = torch::cat({flowBackwardCPU, torch::ones({flowBackwardCPU.sizes()[0], flowBackwardCPU.sizes()[1], 1l})}, 2);
            torch::Tensor flowFrameBackwardCPU = flowFrameBackward.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor flowBackwardDiffCPU = flowBackwardDiff.index({(int)displaySeqId, "..."}).contiguous().to(torch::kCPU);
#endif
            torch::Tensor outputCPU = batchOut.index({(int)displaySeqId, "..."}).to(torch::kCPU);
#if DOUBLE_EDEC
            torch::Tensor output2CPU = batchOut2.index({(int)displaySeqId, "..."}).to(torch::kCPU);
#endif
            torch::Tensor batchOutScaled1CPU = batchOutScaled1.index({(int)displaySeqId, "..."}).to(torch::kCPU);
            torch::Tensor batchOutScaled2CPU = batchOutScaled2.index({(int)displaySeqId, "..."}).to(torch::kCPU);

            _trainingInfo->images["input_1"].write()->copyFrom(batchIn1CPU.data_ptr<float>());
            _trainingInfo->images["input_1_scaled_1"].write()->copyFrom(batchIn1Scaled1CPU.data_ptr<float>());
            _trainingInfo->images["input_1_scaled_2"].write()->copyFrom(batchIn1Scaled2CPU.data_ptr<float>());
            _trainingInfo->images["input_2"].write()->copyFrom(batchIn2CPU.data_ptr<float>());
            _trainingInfo->images["prediction_1"].write()->copyFrom(outputCPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction_1_scaled_1"].write()->copyFrom(batchOutScaled1CPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction_1_scaled_2"].write()->copyFrom(batchOutScaled2CPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["prediction_1_double_edec"].write()->copyFrom(output2CPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["flow_forward"].write()->copyFrom(flowForwardCPU.data_ptr<float>());
            _trainingInfo->images["flow_forward_mapped"].write()->copyFrom(flowFrameForwardCPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["flow_forward_diff"].write()->copyFrom(flowForwardDiffCPU.data_ptr<float>());
            _trainingInfo->images["flow_backward"].write()->copyFrom(flowBackwardCPU.data_ptr<float>());
            _trainingInfo->images["flow_backward_mapped"].write()->copyFrom(flowFrameBackwardCPU.permute({1, 2, 0}).contiguous().data_ptr<float>());
            _trainingInfo->images["flow_backward_diff"].write()->copyFrom(flowBackwardDiffCPU.data_ptr<float>());
        }

        // Print loss
        printf(
            "\r[%2ld/%2ld][%3ld/%3ld] loss: %9.6f frameLoss: %9.6f flowForwardLoss: %9.6f flowBackwardLoss: %9.6f frameLossDoubleEdec: %9.6f doubleEdecLoss: %9.6f encMean: [%9.6f %9.6f] encVar: [%9.6f %9.6f]",
            ti, nTrainingIterations, t, sequenceLength,
            loss.item<float>(), frameLoss.item<float>(),
#if FLOW
            flowForwardLoss.item<float>(), flowBackwardLoss.item<float>(),
#else
            0.0, 0.0,
#endif
#if DOUBLE_EDEC
            frameLossDoubleEdec.item<float>(), doubleEncodingLoss.item<float>(),
#else
            0.0, 0.0,
#endif
            encodingMean.min().item<float>(), encodingMean.max().item<float>(),
            encodingVar.min().item<float>(), encodingVar.max().item<float>());
        fflush(stdout);

        {   // Write the time series
            auto timeSeriesWriteHandle = _trainingInfo->timeSeries.write();
            auto currentTime = high_resolution_clock::now();

            timeSeriesWriteHandle->addEntries(
                "time", (double)duration_cast<milliseconds>(currentTime-_trainingStartTime).count() / 1000.0,
                "loss", loss.item<double>(),
                "frameLoss", frameLoss.item<double>(),
                "flowForwardLoss", flowForwardLoss.item<double>(),
                "flowBackwardLoss", flowBackwardLoss.item<double>(),
                "frameLossDoubleEdec", frameLossDoubleEdec.item<double>(),
                "doubleEncodingLoss", doubleEncodingLoss.item<double>()
            );
        }

        if (_abortTraining)
        {
            break;
        }
    }

    // Save models
    try
    {
        {
            printf("Saving frame encoder model to %s\n", doot2::frameEncoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameEncoder->save(outputArchive);
            outputArchive.save_to(doot2::frameEncoderFilename);
        }
        {
            printf("Saving frame decoder model to %s\n", doot2::frameDecoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameDecoder->save(outputArchive);
            outputArchive.save_to(doot2::frameDecoderFilename);
        }
        {
            printf("Saving flow decoder model to %s\n", doot2::flowDecoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _flowDecoder->save(outputArchive);
            outputArchive.save_to(doot2::flowDecoderFilename);
        }
    }
    catch (const std::exception& e) {
        printf("Could not save the models: '%s'\n", e.what());
    }
}

void AutoEncoderModel::reset()
{

}

void AutoEncoderModel::infer(const TensorVector& input, TensorVector& output)
{
    // TODO
}

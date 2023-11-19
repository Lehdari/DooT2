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
#if 0
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
#endif
    // returns YUV local mean (B*3*H*W), YUV local variance (B*3*H*W), UV local covariance (B*1*H*W)
    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> localVariance(const torch::Tensor& yuvImage)
    {
        using namespace torch::indexing;

        auto device = yuvImage.device();
        auto dtype = yuvImage.dtype();

        // Gaussian kernel
        static float kernelData[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };
        static torch::Tensor kernelV = torch::from_blob(kernelData, {5,1}).repeat({1,1,1,1}).to(device, dtype);
        static torch::Tensor kernelH = torch::from_blob(kernelData, {1,5}).repeat({1,1,1,1}).to(device, dtype);
        static torch::Tensor kernelV3 = torch::from_blob(kernelData, {5,1}).repeat({3,1,1,1}).to(device, dtype);
        static torch::Tensor kernelH3 = torch::from_blob(kernelData, {1,5}).repeat({3,1,1,1}).to(device, dtype);

        torch::Tensor mean = torch::reflection_pad2d(yuvImage, {0,0,2,2});
        mean = tf::conv2d(mean, kernelV3, tf::Conv2dFuncOptions().groups(3));
        mean = torch::reflection_pad2d(mean, {2,2,0,0});
        mean = tf::conv2d(mean, kernelH3, tf::Conv2dFuncOptions().groups(3));

        torch::Tensor var = torch::reflection_pad2d(yuvImage.square(), {0,0,2,2});
        var = tf::conv2d(var, kernelV3, tf::Conv2dFuncOptions().groups(3));
        var = torch::reflection_pad2d(var, {2,2,0,0});
        var = tf::conv2d(var, kernelH3, tf::Conv2dFuncOptions().groups(3));
        var -= mean.square();

        torch::Tensor uvCovar = yuvImage.index({Slice(), Slice(1,2), Slice(), Slice()}) *
            yuvImage.index({Slice(), Slice(2,3), Slice(), Slice()}).to(device, dtype);
        uvCovar = torch::reflection_pad2d(uvCovar, {0,0,2,2}).to(device, dtype);
        uvCovar = tf::conv2d(uvCovar, kernelV, tf::Conv2dFuncOptions()).to(device, dtype);
        uvCovar = torch::reflection_pad2d(uvCovar, {2,2,0,0}).to(device, dtype);
        uvCovar = tf::conv2d(uvCovar, kernelH, tf::Conv2dFuncOptions()).to(device, dtype);
        uvCovar -= (mean.index({Slice(), Slice(1,2), Slice(), Slice()}) *
            mean.index({Slice(), Slice(2,3), Slice(), Slice()})).to(device, dtype);

        return {mean, var, uvCovar};
    }

    // returns Variance Y error, Variance UV error, L2 Y error, L2 UV error
    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        yuvImageErrors(const torch::Tensor& yuvImageTarget, const torch::Tensor& yuvImagePred, bool debug=false)
    {
        using namespace torch::indexing;
        constexpr double eps = 2.0e-4;

        auto device = yuvImageTarget.device();
        auto dtype = yuvImageTarget.dtype();

        if (debug) {
            static double k[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };
            torch::Tensor img = yuvImageTarget.index({0}).to(torch::kCPU, torch::kFloat32);
            double mean[2] = {};
            double var[2] = {};
            double var2[2] = {};
            for (int j=0; j<5; ++j) {
                for (int i=0; i<5; ++i) {
                    for (int c=0; c<2; ++c)
                        mean[c] += k[i]*k[j]*img.index({c+1, j, i}).item<double>();
                }
            }
            for (int j=0; j<5; ++j) {
                for (int i=0; i<5; ++i) {
                    for (int c=0; c<2; ++c)
                        var[c] += k[i]*k[j]*std::pow(img.index({c+1, j, i}).item<double>()-mean[c], 2.0);
                }
            }
            for (int j=0; j<5; ++j) {
                for (int i=0; i<5; ++i) {
                    for (int c=0; c<2; ++c)
                        var2[c] += k[i]*k[j]*std::pow(img.index({c+1, j, i}).item<double>(), 2.0);
                }
            }
            for (int c=0; c<2; ++c)
                var2[c] -= mean[c]*mean[c];
            printf("GT mean: [ %10.7f %10.7f ]^T\n", mean[0], mean[1]);
            printf("GT var:  [ %10.7f %10.7f ]^T\n", var[0], var[1]);
            printf("GT var2: [ %10.7f %10.7f ]^T\n", var2[0], var2[1]);
        }

        auto [mean, var, uvCovar] = localVariance(yuvImageTarget);
        var = (var + eps).to(device, dtype); // Add epsilon to variance for variance error stability
        torch::Tensor diff = (yuvImagePred-yuvImageTarget).to(device, dtype);
        torch::Tensor yVarErr = 1.0 - torch::exp(-0.5 * (
            diff.index({Slice(), Slice(0,1), Slice(), Slice()}).square() /
            (var.index({Slice(), Slice(0,1), Slice(), Slice()}))).to(device, dtype));

        if (debug) {
            printf("uvMean: [ %10.7f %10.7f ]^T\n", mean.index({0,1,2,2}).item<double>(), mean.index({0,2,2,2}).item<double>());
            printf("uvVar: [ %10.7f %10.7f ]^T\n", var.index({0,1,2,2}).item<double>(), var.index({0,2,2,2}).item<double>());
            printf("uvCovar: %10.7f\n", uvCovar.index({0,0,2,2}).item<double>());
        }

        // Construct tensor of inverses of UV covariance matrices
        torch::Tensor uvVarErr = (torch::cat({
            var.index({Slice(), Slice(2,3), Slice(), Slice()}),
            -uvCovar, -uvCovar,
            var.index({Slice(), Slice(1,2), Slice(), Slice()})
        }, 1) * (
        1.0 / (
            var.index({Slice(), Slice(1,2), Slice(), Slice()}) * var.index({Slice(), Slice(2,3), Slice(), Slice()}) -
            uvCovar.square()
        )))
        .reshape({diff.sizes()[0], 2, 2, diff.sizes()[2], diff.sizes()[3]}).permute({0,3,4,1,2}).to(device, dtype);

        if (debug) {
            printf("covar matrix inverse:\n[ %8.5f %8.5f ]\n[ %8.5f %8.5f ]\n",
                uvVarErr.index({0,2,2,0,0}).item<double>(),
                uvVarErr.index({0,2,2,0,1}).item<double>(),
                uvVarErr.index({0,2,2,1,0}).item<double>(),
                uvVarErr.index({0,2,2,1,1}).item<double>()
            );

            {
                double a = var.index({0,1,2,2}).item<double>();
                double d = var.index({0,2,2,2}).item<double>();
                double b = uvCovar.index({0,0,2,2}).item<double>();
                printf("check:\n[ %8.5f %8.5f ]\n[ %8.5f %8.5f ]\n",
                    (1.0/(a*d-b*b)) * d,
                    (1.0/(a*d-b*b)) * -b,
                    (1.0/(a*d-b*b)) * -b,
                    (1.0/(a*d-b*b)) * a
                );
            }
        }

        // diff^T * sigma^-1 * diff
        torch::Tensor uvDiff = diff.index({Slice(), Slice(1,3), Slice(), Slice()}).permute({0,2,3,1}).to(device, dtype);
        uvVarErr = torch::matmul(uvVarErr, uvDiff.unsqueeze(-1)).squeeze(-1).to(device, dtype);
        uvVarErr = torch::linalg_vecdot(uvDiff, uvVarErr).unsqueeze(-1).permute({0,3,1,2}).to(device, dtype);
        if (debug) {
            printf("uvDiff: [ %8.5f %8.5f ]^T\n", diff.index({0,1,2,2}).item<double>(), diff.index({0,2,2,2}).item<double>());
            printf("diff^T * sigma^-1 * diff: %8.5f\n", uvVarErr.index({0,0,2,2}).item<double>());
        }

        // final error
        uvVarErr = (1.0 - torch::exp(-0.5 * uvVarErr)).to(device, dtype);

        return {
            yVarErr,
            uvVarErr,
            diff.index({Slice(), Slice(0,1), Slice(), Slice()}).square().to(device, dtype),
            (diff.index({Slice(), Slice(1,3), Slice(), Slice()}).square().sum(1, true) * 4.0).to(device, dtype)
        };
    }

    inline torch::Tensor yuvImageCombinedError(
        const torch::Tensor& yuvImageTarget,
        const torch::Tensor& yuvImagePred,
        double yVarErrorWeight = 1.0,
        double uvVarErrorWeight = 0.5,
        double yL2ErrorWeight = 50.0,
        double uvL2ErrorWeight = 15.0
        )
    {
        auto device = yuvImageTarget.device();
        auto dtype = yuvImageTarget.dtype();

        auto [yVarError, uvVarError, yL2Error, uvL2Error] = yuvImageErrors(yuvImageTarget, yuvImagePred);
        return (yVarError*yVarErrorWeight +
            uvVarError*uvVarErrorWeight +
            yL2Error*yL2ErrorWeight +
            uvL2Error*uvL2ErrorWeight).to(device, dtype);
    }

    inline torch::Tensor yuvImageLoss(
        const torch::Tensor& yuvImageTarget,
        const torch::Tensor& yuvImagePred,
        double yVarErrorWeight = 1.0,
        double uvVarErrorWeight = 0.5,
        double yL2ErrorWeight = 50.0,
        double uvL2ErrorWeight = 15.0)
    {
        auto device = yuvImageTarget.device();
        auto dtype = yuvImageTarget.dtype();

        torch::Tensor combinerErrorImage = yuvImageCombinedError(yuvImageTarget, yuvImagePred,
            yVarErrorWeight, uvVarErrorWeight, yL2ErrorWeight, uvL2ErrorWeight);
        return combinerErrorImage.mean();
    }

    inline std::tuple<torch::Tensor, torch::Tensor> imageSpectrum(const torch::Tensor& image)
    {
        torch::Tensor spectrum = torch::fft_rfft2(image.to(torch::kFloat32)); // [B * C/2 * H * W/2] complex tensor, C: _globalInputChannels
        torch::Tensor amp = spectrum.abs();
        torch::Tensor phase = spectrum.angle();

        return { amp, phase };
    }

    inline std::tuple<torch::Tensor, torch::Tensor> imageSpectrumErrors(
        const torch::Tensor& imageTarget, const torch::Tensor& imagePred)
    {
        torch::Tensor spectrumTarget = torch::fft_rfft2(imageTarget.to(torch::kFloat32)); // [B * C/2 * H * W/2] complex tensor, C: _globalInputChannels
        torch::Tensor spectrumPred = torch::fft_rfft2(imagePred.to(torch::kFloat32)); // [B * C/2 * H * W/2] complex tensor, C: _globalInputChannels

        constexpr float eps = 1.0e-6;

        torch::Tensor targetAbs = spectrumTarget.abs();
        targetAbs = torch::max(targetAbs, eps*torch::ones_like(targetAbs));
        torch::Tensor predAbs = spectrumPred.abs();
        predAbs = torch::max(predAbs, eps*torch::ones_like(predAbs));

        spectrumTarget = torch::view_as_real(spectrumTarget);
        spectrumPred = torch::view_as_real(spectrumPred);

        torch::Tensor ampLoss = (predAbs - targetAbs).abs();
        torch::Tensor phaseDiff = M_1_PI*acos(clamp((
            (spectrumTarget / targetAbs.unsqueeze(-1)) * // division by abs for normalization
            (spectrumPred / predAbs.unsqueeze(-1))
            ).sum(-1),
            -1.0+eps, 1.0-eps));
        // filtering for phase loss discontinuity at origin
        torch::Tensor spectrumTargetDot = (spectrumTarget * spectrumTarget).sum(-1);
        spectrumTargetDot = torch::max(spectrumTargetDot, eps*torch::ones_like(spectrumTargetDot));
        torch::Tensor dirDiff = 0.5 - 0.5*((spectrumPred * spectrumTarget).sum(-1) / spectrumTargetDot);
        torch::Tensor a = clamp(1.0 - predAbs / targetAbs, 0.0, 1.0).square();
//        printf("a: [ %d %d %d %d ]\n", a.sizes()[0], a.sizes()[1], a.sizes()[2], a.sizes()[3]);
//        printf("phaseDiff: [ %d %d %d %d ]\n", phaseDiff.sizes()[0], phaseDiff.sizes()[1], phaseDiff.sizes()[2], phaseDiff.sizes()[3]);
//        printf("dirDiff: [ %d %d %d %d ]\n", dirDiff.sizes()[0], dirDiff.sizes()[1], dirDiff.sizes()[2], dirDiff.sizes()[3]);
//        printf("targetAbs: [ %d %d %d %d ]\n", targetAbs.sizes()[0], targetAbs.sizes()[1], targetAbs.sizes()[2], targetAbs.sizes()[3]);
//        fflush(stdout);
        torch::Tensor phaseLoss = ((1.0-a)*phaseDiff + a*dirDiff) * targetAbs;

        return { ampLoss, phaseLoss };
//        printf("targetAbs: [%d %d %d %d]\n",
//            targetAbs.sizes()[0], targetAbs.sizes()[1], targetAbs.sizes()[2], targetAbs.sizes()[3]);
//        fflush(stdout);

        // Amplitude error is L2 loss for log amplitudes
//        torch::Tensor errAmp = torch::abs(targetAbs - predAbs);

//        printf("spectrumTarget.abs().max(): %0.5f\n", spectrumTarget.abs().max().item<double>());
//        printf("spectrumPred.abs().max(): %0.5f\n", spectrumPred.abs().max().item<double>());

        // Phase error based on dot products on the complex plane
//        torch::Tensor errPhase = torch::acos(torch::clamp((
//            (torch::view_as_real(spectrumTarget) / targetAbs.unsqueeze(-1)) * // division by abs for normalization
//            (torch::view_as_real(spectrumPred) / predAbs.unsqueeze(-1))
//            ).sum(-1), -0.9999, 0.9999));
//
//        return { errAmp, errPhase };
    }

    inline torch::Tensor imageSpectrumLoss(
        const torch::Tensor& imageTarget, const torch::Tensor& imagePred)
    {
        auto [errAmp, errPhase] = imageSpectrumErrors(imageTarget, imagePred);

//        printf("errAmp: %0.5f errPhase: %0.5f\n", errAmp.mean().item<double>(), errPhase.mean().item<double>());

        return errAmp.mean() + errPhase.mean();
    }

    INLINE double distributionLossKernelWidth(double batchSize)
    {
        // Function fitted to kernel widths acquired by empirical tests
        return 0.35443676461512874 + 1.404941003999142/(0.31916811423436164*std::pow(batchSize+8.424505177725445,
            0.8104176806922638));
    }

    INLINE torch::Tensor circularLoss(const torch::Tensor& x, const torch::Tensor& y)
    {
        torch::Tensor xNorm = torch::linalg_norm(x, c10::nullopt, 1);
        torch::Tensor yNorm = torch::linalg_norm(y, c10::nullopt, 1);
        torch::Tensor xn = torch::div(x, xNorm.unsqueeze(1).expand({-1, doot2::encodingLength}));
        torch::Tensor yn = torch::div(y, yNorm.unsqueeze(1).expand({-1, doot2::encodingLength}));
        return torch::mean(
            1.0-torch::sum(torch::mul(xn, yn), 1) + // cosine loss
            (xNorm-yNorm).square() / constexprSqrt(doot2::encodingLength)); // distance loss
    }

    INLINE torch::Tensor bathtubLoss(const torch::Tensor& x, double alpha)
    {
        double a = alpha*alpha / (2.0*alpha*alpha - 2.0*alpha + 1.0);
        double b = std::sqrt(a*(1.0-a));
        double c = 1.0 / (3.61512 - 1.96*b);
        return torch::mean(c * (a*(1/x-1) + (1-a)*(1/(1-x)-1) - 2.0*b));
    }

    INLINE torch::Tensor createHannWindow(const std::vector<long>& size, torch::Device device, double eps=0.1)
    {
        auto m = torch::meshgrid({
            torch::linspace(0.0, 2.0*M_PI, size[0]),
            torch::linspace(0.0, 2.0*M_PI, size[1])}, "ij");
        m[0] = 0.5-0.5*torch::cos(m[0]);
        m[1] = 0.5-0.5*torch::cos(m[1]);
        return ((m[0]*m[1])*(1.0-eps) + eps).view({1,1,size[0],size[1]}).to(device);
    }

} // namespace

Json MultiLevelAutoEncoderModel::getDefaultModelConfig()
{
    Json modelConfig;

    modelConfig["frame_encoder_filename"] = "frame_encoder.pt";
    modelConfig["frame_decoder_filename"] = "frame_decoder.pt";
    modelConfig["discriminator_filename"] = "discriminator.pt";
    modelConfig["encoding_discriminator_filename"] = "encoding_discriminator.pt";
    modelConfig["frame_classifier_filename"] = "frame_classifier.pt";
    modelConfig["optimizer_state_filename"] = "optimizer_state.pt";
    modelConfig["optimizer_learning_rate"] = 0.001;
    modelConfig["optimizer_beta1"] = 0.9;
    modelConfig["optimizer_beta2"] = 0.999;
    modelConfig["optimizer_epsilon"] = 1.0e-8;
    modelConfig["optimizer_weight_decay"] = 0.05;
    modelConfig["optimizer_learning_rate_initial"] = 0.0001;
    modelConfig["optimizer_beta1_initial"] = 0.0;
    modelConfig["optimizer_weight_decay_initial"] = 0.0;
    modelConfig["optimizer_learning_rate_final"] = 0.001;
    modelConfig["optimizer_beta1_final"] = 0.9;
    modelConfig["optimizer_weight_decay_final"] = 0.05;
    modelConfig["warmup_duration"] = 2048;
    modelConfig["n_training_cycles"] = 1;
    modelConfig["virtual_batch_size"] = 256;
    modelConfig["frame_loss_weight"] = 4.0;
    modelConfig["frame_grad_loss_weight"] = 1.5;
    modelConfig["frame_laplacian_loss_weight"] = 1.5;
    modelConfig["use_frame_classification_loss"] = false;
    modelConfig["frame_classification_loss_weight"] = 0.3;
    modelConfig["use_encoding_mean_loss"] = false;
    modelConfig["encoding_mean_loss_weight"] = 0.001;
    modelConfig["use_encoding_distribution_loss"] = false;
    modelConfig["encoding_distribution_loss_weight"] = 100.0;
    modelConfig["use_encoding_distance_loss"] = false;
    modelConfig["encoding_distance_loss_weight"] = 0.001;
    modelConfig["use_encoding_covariance_loss"] = false;
    modelConfig["encoding_covariance_loss_weight"] = 0.02;
    modelConfig["use_encoding_prev_distance_loss"] = false;
    modelConfig["encoding_prev_distance_loss_weight"] = 0.4;
    modelConfig["use_encoding_discrimination_loss"] = false;
    modelConfig["encoding_discrimination_loss_weight"] = 0.3;
    modelConfig["use_encoding_circular_loss"] = false;
    modelConfig["encoding_circular_loss_weight"] = 0.25;
    modelConfig["use_discriminator"] = false;
    modelConfig["discrimination_loss_weight"] = 0.4;
    modelConfig["discriminator_virtual_batch_size"] = 8;
    modelConfig["target_reconstruction_loss"] = 0.1;

    return modelConfig;
}

MultiLevelAutoEncoderModel::MultiLevelAutoEncoderModel() :
    _optimizer                          (std::make_unique<torch::optim::AdamW>(
                                         std::vector<optim::OptimizerParamGroup>
                                         {_frameEncoder->parameters(), _frameDecoder->parameters()})),
    _discriminatorOptimizer             (std::make_unique<torch::optim::AdamW>(
                                         std::vector<optim::OptimizerParamGroup>
                                         {_discriminator->parameters()})),
    _device                             (torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    _optimizerLearningRate              (0.001),
    _optimizerBeta1                     (0.9),
    _optimizerBeta2                     (0.999),
    _optimizerEpsilon                   (1.0e-8),
    _optimizerWeightDecay               (0.05),
    _optimizerLearningRateInitial       (0.0001),
    _optimizerBeta1Initial              (0.0),
    _optimizerWeightDecayInitial        (0.0),
    _optimizerLearningRateFinal         (0.001),
    _optimizerBeta1Final                (0.9),
    _optimizerWeightDecayFinal          (0.05),
    _warmupDuration                     (2048),
    _nTrainingCycles                    (1),
    _virtualBatchSize                   (256),
    _frameLossWeight                    (3.0),
    _frameGradLossWeight                (1.5),
    _frameLaplacianLossWeight           (1.5),
    _useFrameClassificationLoss         (false),
    _frameClassificationLossWeight      (0.3),
    _useEncodingMeanLoss                (false),
    _encodingMeanLossWeight             (0.001),
    _useEncodingDistributionLoss        (false),
    _encodingDistributionLossWeight     (100.0),
    _useEncodingDistanceLoss            (false),
    _encodingDistanceLossWeight         (0.001),
    _useEncodingCovarianceLoss          (false),
    _encodingCovarianceLossWeight       (0.02),
    _useEncodingPrevDistanceLoss        (false),
    _encodingPrevDistanceLossWeight     (0.4),
    _useEncodingDiscriminationLoss      (false),
    _encodingDiscriminationLossWeight   (0.4),
    _useEncodingCircularLoss            (false),
    _encodingCircularLossWeight         (0.25),
    _useDiscriminator                   (false),
    _discriminationLossWeight           (0.4),
    _discriminatorVirtualBatchSize      (8),
    _spectrumLossFiltering              (0.99),
    _spectrumLossWeight                 (1.0),
    _targetReconstructionLoss           (0.1),
    _trainingIteration                  (0),
    _lossLevel                          (0.0),
    _batchPixelDiff                     (1.0),
    _batchEncDiff                       (sqrt(doot2::encodingLength)),
    _frameEncoder                       (4)
{
}

void MultiLevelAutoEncoderModel::init(const Json& experimentConfig)
{
    auto& modelConfig = experimentConfig["model_config"];
    _experimentRoot = doot2::experimentsDirectory / experimentConfig["experiment_root"].get<fs::path>();

    // Setup hyperparameters
    _optimizerLearningRate = modelConfig["optimizer_learning_rate"];
    _optimizerBeta1 = modelConfig["optimizer_beta1"];
    _optimizerBeta2 = modelConfig["optimizer_beta2"];
    _optimizerEpsilon = modelConfig["optimizer_epsilon"];
    _optimizerWeightDecay = modelConfig["optimizer_weight_decay"];
    _optimizerLearningRateInitial = modelConfig["optimizer_learning_rate_initial"];
    _optimizerBeta1Initial = modelConfig["optimizer_beta1_initial"];
    _optimizerWeightDecayInitial = modelConfig["optimizer_weight_decay_initial"];
    _optimizerLearningRateFinal = modelConfig["optimizer_learning_rate_final"];
    _optimizerBeta1Final = modelConfig["optimizer_beta1_final"];
    _optimizerWeightDecayFinal = modelConfig["optimizer_weight_decay_final"];
    _warmupDuration = modelConfig["warmup_duration"];
    _nTrainingCycles = modelConfig["n_training_cycles"];
    _virtualBatchSize = modelConfig["virtual_batch_size"];
    _frameLossWeight = modelConfig["frame_loss_weight"];
    _frameGradLossWeight = modelConfig["frame_grad_loss_weight"];
    _frameLaplacianLossWeight = modelConfig["frame_laplacian_loss_weight"];
    _useFrameClassificationLoss = modelConfig["use_frame_classification_loss"];
    _frameClassificationLossWeight = modelConfig["frame_classification_loss_weight"];
    _useEncodingMeanLoss = modelConfig["use_encoding_mean_loss"];
    _encodingMeanLossWeight = modelConfig["encoding_mean_loss_weight"];
    _useEncodingDistributionLoss = modelConfig["use_encoding_distribution_loss"];
    _encodingDistributionLossWeight = modelConfig["encoding_distribution_loss_weight"];
    _useEncodingDistanceLoss = modelConfig["use_encoding_distance_loss"];
    _encodingDistanceLossWeight = modelConfig["encoding_distance_loss_weight"];
    _useEncodingCovarianceLoss = modelConfig["use_encoding_covariance_loss"];
    _encodingCovarianceLossWeight = modelConfig["encoding_covariance_loss_weight"];
    _useEncodingPrevDistanceLoss = modelConfig["use_encoding_prev_distance_loss"];
    _encodingPrevDistanceLossWeight = modelConfig["encoding_prev_distance_loss_weight"];
    _useEncodingDiscriminationLoss = modelConfig["use_encoding_discrimination_loss"];
    _encodingDiscriminationLossWeight = modelConfig["encoding_discrimination_loss_weight"];
    _useEncodingCircularLoss = modelConfig["use_encoding_circular_loss"];
    _encodingCircularLossWeight = modelConfig["encoding_circular_loss_weight"];
    _useDiscriminator = modelConfig["use_discriminator"];
    _discriminationLossWeight = modelConfig["discrimination_loss_weight"];
    _discriminatorVirtualBatchSize = modelConfig["discriminator_virtual_batch_size"];
    _targetReconstructionLoss = modelConfig["target_reconstruction_loss"];

    // Setup the state parameters
    _trainingIteration = 0;
    _lossLevel = 0.0;
    _lossLevelErrorIntegrate = 0.0;
    _lossLevelErrorFiltered = 0.0;
    _batchPixelDiff = 1.0;
    _batchEncDiff = sqrt(doot2::encodingLength);

    // Load torch model file names from the model config
    _frameEncoderFilename = "frame_encoder.pt";
    if (modelConfig.contains("frame_encoder_filename"))
        _frameEncoderFilename = modelConfig["frame_encoder_filename"].get<fs::path>();

    _frameDecoderFilename = "frame_decoder.pt";
    if (modelConfig.contains("frame_decoder_filename"))
        _frameDecoderFilename = modelConfig["frame_decoder_filename"].get<fs::path>();

    _discriminatorFilename = "discriminator.pt";
    if (modelConfig.contains("discriminator_filename"))
        _discriminatorFilename = modelConfig["discriminator_filename"].get<fs::path>();

    _encodingDiscriminatorFilename = "encoding_discriminator.pt";
    if (modelConfig.contains("encoding_discriminator_filename"))
        _encodingDiscriminatorFilename = modelConfig["encoding_discriminator_filename"].get<fs::path>();

    _frameClassifierFilename = "frame_classifier.pt";
    if (modelConfig.contains("frame_classifier_filename"))
        _frameClassifierFilename = modelConfig["frame_classifier_filename"].get<fs::path>();

    _optimizerStateFilename = "optimizer_state.pt";
    if (modelConfig.contains("optimizer_state_filename"))
        _optimizerStateFilename = modelConfig["optimizer_state_filename"].get<fs::path>();

    // Separate loading paths in case a base experiment is specified
    fs::path frameEncoderFilename = _frameEncoderFilename;
    fs::path frameDecoderFilename = _frameDecoderFilename;
    fs::path discriminatorFilename = _discriminatorFilename;
    fs::path encodingDiscriminatorFilename = _encodingDiscriminatorFilename;
    fs::path frameClassifierFilename = _frameClassifierFilename;
    fs::path optimizerStateFilename = _optimizerStateFilename;
    if (experimentConfig.contains("experiment_base_root") && experimentConfig.contains("base_model_config")) {
        fs::path experimentBaseRoot = experimentConfig["experiment_base_root"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("frame_encoder_filename"))
            frameEncoderFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["frame_encoder_filename"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("frame_decoder_filename"))
            frameDecoderFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["frame_decoder_filename"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("discriminator_filename"))
            discriminatorFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["discriminator_filename"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("encoding_discriminator_filename"))
            encodingDiscriminatorFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["encoding_discriminator_filename"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("frame_classifier_filename"))
            frameClassifierFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["frame_classifier_filename"].get<fs::path>();
        if (experimentConfig["base_model_config"].contains("optimizer_state_filename"))
            optimizerStateFilename = experimentBaseRoot /
                experimentConfig["base_model_config"]["optimizer_state_filename"].get<fs::path>();

        // Load state parameters from the base experiment
        fs::path stateParamsFilename = experimentBaseRoot / "state_params.json";
        std::ifstream stateParamsFile(stateParamsFilename);
        auto stateParams = Json::parse(stateParamsFile);

        if (stateParams.contains("trainingIteration")) _trainingIteration = stateParams["trainingIteration"];
        if (stateParams.contains("lossLevel")) _lossLevel = stateParams["lossLevel"];
        if (stateParams.contains("lossLevelErrorIntegrate")) _lossLevelErrorIntegrate =
            stateParams["lossLevelErrorIntegrate"];
        if (stateParams.contains("lossLevelErrorFiltered")) _lossLevelErrorFiltered =
            stateParams["lossLevelErrorFiltered"];
        if (stateParams.contains("batchPixelDiff")) _batchPixelDiff = stateParams["batchPixelDiff"];
        if (stateParams.contains("batchEncDiff")) _batchEncDiff = stateParams["batchEncDiff"];
        if (stateParams.contains("spectrumLossWeight")) _spectrumLossWeight = stateParams["spectrumLossWeight"];
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
        *_frameEncoder = MultiLevelFrameEncoderImpl(4);
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
            printf("Loading discriminator model from %s\n", discriminatorFilename.c_str()); // TODO logging
            serialize::InputArchive inputArchive;
            inputArchive.load_from(discriminatorFilename);
            _discriminator->load(inputArchive);
        } else {
            printf("No %s found. Initializing new discriminator model.\n",
                discriminatorFilename.c_str()); // TODO logging
            *_discriminator = DiscriminatorImpl();
        }
    }

    // Load encoding discriminator
    if (fs::exists(encodingDiscriminatorFilename)) {
        printf("Loading encoding discriminator model from %s\n", encodingDiscriminatorFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(encodingDiscriminatorFilename);
        _encodingDiscriminator->load(inputArchive);
    } else {
        printf("No %s found. Initializing new encoding discriminator model.\n",
            encodingDiscriminatorFilename.c_str()); // TODO logging
        *_encodingDiscriminator = EncodingDiscriminatorImpl();
    }

    // Load frame classifier
    if (fs::exists(frameClassifierFilename)) {
        printf("Loading frame classifier model from %s\n", frameClassifierFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameClassifierFilename);
        _frameClassifier->load(inputArchive);
    } else {
        printf("No %s found. Initializing new frame classifier model.\n",
            frameClassifierFilename.c_str()); // TODO logging
        *_frameClassifier = EncodingDiscriminatorImpl();
    }

    // Move model parameters to GPU if it's used
    _frameEncoder->to(_device, torch::kBFloat16);
    _frameDecoder->to(_device, torch::kBFloat16);
    _discriminator->to(_device, torch::kBFloat16);
    _encodingDiscriminator->to(_device, torch::kBFloat16);
    _frameClassifier->to(_device, torch::kBFloat16);

    // Setup optimizers
    _optimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_frameEncoder->parameters(), _frameDecoder->parameters()});
    if (fs::exists(optimizerStateFilename)) {
        printf("Loading optimizer state from %s\n", optimizerStateFilename.c_str()); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(optimizerStateFilename);
        _optimizer->load(inputArchive);
    }
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

    _encodingDiscriminatorOptimizer = std::make_unique<torch::optim::AdamW>(std::vector<optim::OptimizerParamGroup>
        {_encodingDiscriminator->parameters(), _frameClassifier->parameters()});
    dynamic_cast<torch::optim::AdamWOptions&>(_optimizer->param_groups()[0].options())
        .lr(_optimizerLearningRate)
        .betas({_optimizerBeta1, _optimizerBeta2})
        .eps(_optimizerEpsilon)
        .weight_decay(_optimizerWeightDecay);

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
        timeSeriesWriteHandle->addSeries<double>("frameClassificationLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("frameClassifierLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingDistributionLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingDistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingMeanLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCovarianceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingPrevDistanceLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingDiscriminationLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingDiscriminatorLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminationLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("discriminatorLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingCircularLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("encodingMaskLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("spectrumLoss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("spectrumLossWeight", 0.0);
        timeSeriesWriteHandle->addSeries<double>("reconstructionLosses", 0.0);
        timeSeriesWriteHandle->addSeries<double>("auxiliaryLosses", 0.0);
        timeSeriesWriteHandle->addSeries<double>("loss", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevel", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevelErrorIntegrate", 0.0);
        timeSeriesWriteHandle->addSeries<double>("lossLevelErrorFiltered", 0.0);
        timeSeriesWriteHandle->addSeries<double>("controlP", 0.0);
        timeSeriesWriteHandle->addSeries<double>("controlI", 0.0);
        timeSeriesWriteHandle->addSeries<double>("controlD", 0.0);
        timeSeriesWriteHandle->addSeries<double>("optimizerLearningRate", 0.0);
        timeSeriesWriteHandle->addSeries<double>("optimizerBeta1", 0.0);
        timeSeriesWriteHandle->addSeries<double>("optimizerWeightDecay", 0.0);
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
        // TODO temp
        for (auto i=0; i<7; ++i) {
            *(_trainingInfo->images)["uvCovar" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["varErrorY" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["varErrorUV" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["l2ErrorY" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["l2ErrorUV" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["totalLoss" + std::to_string(i)].write() = Image<float>(width, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumAmp" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumPhase" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumAmpPrediction" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumPhasePrediction" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumAmpError" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
            *(_trainingInfo->images)["spectrumPhaseError" + std::to_string(i)].write() = Image<float>(width/2, height, ImageFormat::GRAY);
        }
        // TODO end of temp
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
        *(_trainingInfo->images)["encoding_mask"].write() = Image<float>(64*8, 32*8, ImageFormat::GRAY);
        *(_trainingInfo->images)["random_encoding"].write() = Image<float>(64*8, 32*8, ImageFormat::GRAY);
        if (_useEncodingCovarianceLoss) {
            *(_trainingInfo->images)["covariance_matrix"].write() = Image<float>(
                doot2::encodingLength, doot2::encodingLength, ImageFormat::GRAY);
        }
    }
}

void MultiLevelAutoEncoderModel::save(const fs::path& subdir)
{
    try {
        // Save the state parameters
        {
            Json stateParams;
            stateParams["trainingIteration"] = _trainingIteration;
            stateParams["lossLevel"] = _lossLevel;
            stateParams["lossLevelErrorIntegrate"] = _lossLevelErrorIntegrate;
            stateParams["lossLevelErrorFiltered"] = _lossLevelErrorFiltered;
            stateParams["batchPixelDiff"] = _batchPixelDiff;
            stateParams["batchEncDiff"] = _batchEncDiff;
            stateParams["spectrumLossWeight"] = _spectrumLossWeight;

            fs::path stateParamsFilename = _experimentRoot / subdir / "state_params.json";
            printf("Saving model state parameters to %s\n", stateParamsFilename.c_str());
            std::ofstream stateParamsFile(stateParamsFilename);
            stateParamsFile << std::setw(4) << stateParams;
        }

        // Save the models
        {
            fs::path frameEncoderFilename = _experimentRoot / subdir / _frameEncoderFilename;
            printf("Saving frame encoder model to %s\n", frameEncoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameEncoder->save(outputArchive);
            outputArchive.save_to(frameEncoderFilename);
        }
        {
            fs::path frameDecoderFilename = _experimentRoot / subdir / _frameDecoderFilename;
            printf("Saving frame decoder model to %s\n", frameDecoderFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameDecoder->save(outputArchive);
            outputArchive.save_to(frameDecoderFilename);
        }
        {
            fs::path discriminatorFilename = _experimentRoot / subdir / _discriminatorFilename;
            printf("Saving discriminator model to %s\n", discriminatorFilename.c_str());
            serialize::OutputArchive outputArchive;
            _discriminator->save(outputArchive);
            outputArchive.save_to(discriminatorFilename);
        }
        {
            fs::path encodingDiscriminatorFilename = _experimentRoot / subdir / _encodingDiscriminatorFilename;
            printf("Saving encoding discriminator model to %s\n", encodingDiscriminatorFilename.c_str());
            serialize::OutputArchive outputArchive;
            _encodingDiscriminator->save(outputArchive);
            outputArchive.save_to(encodingDiscriminatorFilename);
        }
        {
            fs::path frameClassifierFilename = _experimentRoot / subdir / _frameClassifierFilename;
            printf("Saving frame classifier model to %s\n", frameClassifierFilename.c_str());
            serialize::OutputArchive outputArchive;
            _frameClassifier->save(outputArchive);
            outputArchive.save_to(frameClassifierFilename);
        }

        // Save the optimizer state
        {
            fs::path optimizerStateFilename = _experimentRoot / subdir / _optimizerStateFilename;
            printf("Saving optimizer state to %s\n", optimizerStateFilename.c_str());
            serialize::OutputArchive outputArchive;
            _optimizer->save(outputArchive);
            outputArchive.save_to(optimizerStateFilename);
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
    auto [enc, mask] = _frameEncoder->forward(in);

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

    auto inputDevice = input[0].device();
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
        .to(inputDevice));
    output.push_back(enc.to(inputDevice));
    output.push_back(mask.to(inputDevice));
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
    auto seq = scaleSequences(storage);
#if 0
    // Random sample the pixel diff to get an approximation
    if (_useEncodingDistanceLoss) {
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
            torch::Tensor enc = std::get<0>(_frameEncoder(in));
            _batchEncDiff = _batchEncDiff*0.95 + 0.05*torch::norm(enc.index({b1})-enc.index({b2})).item<double>();
        }
    }
#endif
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

    torch::Tensor zero = torch::zeros({}, TensorOptions().device(_device));

    // Target covariance matrix
    torch::Tensor targetCovarianceMatrix;
    if (_useEncodingCovarianceLoss)
        targetCovarianceMatrix = torch::eye(doot2::encodingLength, TensorOptions().device(_device));

    double lossAcc = 0.0; // accumulate loss over the optimization interval to compute average
    double frameLossAcc = 0.0;
    double frameGradLossAcc = 0.0;
    double frameLaplacianLossAcc = 0.0;
    double frameClassificationLossAcc = 0.0;
    double frameClassifierLossAcc = 0.0;
    double encodingDistributionLossAcc = 0.0;
    double encodingDistanceLossAcc = 0.0;
    double encodingMeanLossAcc = 0.0;
    double encodingCovarianceLossAcc = 0.0;
    double encodingPrevDistanceLossAcc = 0.0;
    double encodingDiscriminationLossAcc = 0.0;
    double encodingDiscriminatorLossAcc = 0.0;
    double discriminationLossAcc = 0.0;
    double discriminatorLossAcc = 0.0;
    double encodingCircularLossAcc = 0.0;
    double encodingMaskLossAcc = 0.0;
    double spectrumLossAcc = 0.0;
    double reconstructionLossesAcc = 0.0;
    double auxiliaryLossesAcc = 0.0;
    double maxEncodingCovariance = 1.0; // max value in the covariance matrix
    int64_t nVirtualBatchesPerCycle = (int64_t)sequenceLength / _virtualBatchSize;
    const double framesPerCycle = (double)nVirtualBatchesPerCycle * _virtualBatchSize;
    for (int64_t c=0; c<_nTrainingCycles; ++c) {
        for (int64_t v=0; v<nVirtualBatchesPerCycle; ++v) {
            MultiLevelImage rOut;
            torch::Tensor covarianceMatrix, enc, encPrev, encRandom, encCircular;
#if 0
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
                            torch::zeros({16, 3, 480, 640}, TensorOptions().device(_device)),
                            //seq.img7.index({(int) t}),
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
#endif
            int displayFrameId = rnd() % _virtualBatchSize;
            for (int64_t b=0; b<_virtualBatchSize; ++b) {
                // frame (time point) to use this iteration
                int t = v*_virtualBatchSize + b;

                // Prepare the inputs (original and scaled)
#if 0
                MultiLevelImage in {
                    seq.img0.index({(int)t}),
                    seq.img1.index({(int)t}),
                    seq.img2.index({(int)t}),
                    seq.img3.index({(int)t}),
                    seq.img4.index({(int)t}),
                    seq.img5.index({(int)t}),
                    seq.img6.index({(int)t}),
                    torch::zeros({16, 3, 480, 640}, TensorOptions().device(_device)),
                    //seq.img7.index({(int) t}),
                    _lossLevel
                };
#endif
                seq.level = _lossLevel;

                // Frame encode
                torch::Tensor encMask;
                std::tie(enc, encMask) = _frameEncoder->forwardSeq(seq, t);

                // Compute distance from encoding mean to origin and encoding mean loss
                torch::Tensor encodingMeanLoss = zero;
                torch::Tensor encMean;
                if (_useEncodingMeanLoss or _useEncodingCovarianceLoss)
                    encMean = enc.mean(0);
                if (_useEncodingMeanLoss) {
                    encodingMeanLoss = encMean.square().sum() * _encodingMeanLossWeight;
                    encodingMeanLossAcc += encodingMeanLoss.item<double>();
                }

                // Compute distribution loss (demand that each encoding channel follows standard normal distribution)
                torch::Tensor encodingDistributionLoss = zero;
                if (_useEncodingDistributionLoss) {
                    torch::Tensor x = torch::linspace(-5.0, 5.0, 64, TensorOptions().device(_device));
                    torch::Tensor mean = enc.expand({x.sizes()[0], enc.sizes()[0], enc.sizes()[1]});
                    torch::Tensor y = normalDistribution(mean,
                        x.unsqueeze(1).unsqueeze(1).expand({x.sizes()[0], enc.sizes()[0], enc.sizes()[1]}),
                        distributionLossKernelWidth(doot2::batchSize)
                    ).mean(1);
                    torch::Tensor y2 = standardNormalDistribution(
                        x.unsqueeze(1).expand({x.sizes()[0], enc.sizes()[1]}));

                    encodingDistributionLoss = _encodingDistributionLossWeight*torch::mse_loss(y, y2);
                    encodingDistributionLossAcc += encodingDistributionLoss.item<double>();
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
#if 0
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
#endif
                encodingPrevDistanceLossAcc += encodingPrevDistanceLoss.item<double>();

                // Encoding mask loss
                torch::Tensor encodingMaskLoss = zero;
                /* if (_useEncodingMaskLoss) */{
                    double _encodingMaskLossWeight = 0.1*std::pow(0.65, _lossLevel);
                    double maskTarget = 0.3 + std::clamp(_lossLevel / 5.0, 0.0, 1.0) * 0.6; // from 0.3 to 0.9
                    encodingMaskLoss = _encodingMaskLossWeight*
                        bathtubLoss(0.001+encMask.to(torch::kFloat32)*0.998, maskTarget).to(encMask.dtype());
                }
                encodingMaskLossAcc += encodingMaskLoss.item<double>();

                // stop gradient from flowing to the previous iteration
                encPrev = enc.detach();

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

                if (_useDiscriminator || _useEncodingCircularLoss || _useEncodingDiscriminationLoss) {
                    encRandom = torch::randn({doot2::batchSize, doot2::encodingLength},
                        TensorOptions().device(_device).dtype(torch::kBFloat16));
                    rOut = _frameDecoder(encRandom, _lossLevel);
                }
                if (_useEncodingCircularLoss || _useEncodingDiscriminationLoss) {
                    encCircular = std::get<0>(_frameEncoder(rOut));
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
                    encodingCircularLoss = _encodingCircularLossWeight*circularLoss(encCircular, encRandom);
                }
                encodingCircularLossAcc += encodingCircularLoss.item<double>();

                // Encoding discrimination loss
                torch::Tensor encodingDiscriminationLoss = zero;
                if (_useEncodingDiscriminationLoss) {
                    torch::Tensor encodingDiscrimination = _encodingDiscriminator(enc);
                    encodingDiscriminationLoss = _encodingDiscriminationLossWeight*
                        torch::binary_cross_entropy_with_logits(encodingDiscrimination,
                            torch::ones_like(encodingDiscrimination));
                    encodingDiscrimination = _encodingDiscriminator(encCircular);
                    encodingDiscriminationLoss += _encodingDiscriminationLossWeight*
                        torch::binary_cross_entropy_with_logits(encodingDiscrimination,
                            torch::ones_like(encodingDiscrimination));
                }
                encodingDiscriminationLossAcc += encodingDiscriminationLoss.item<double>();

                // Frame classification loss
                torch::Tensor frameClassificationLoss = zero;
                if (_useFrameClassificationLoss) {
                    torch::Tensor classification = _frameClassifier(encCircular);
                    frameClassificationLoss = _frameClassificationLossWeight*
                        torch::binary_cross_entropy_with_logits(classification, torch::ones_like(classification));
                }
                frameClassificationLossAcc += frameClassificationLoss.item<double>();

                at::autocast::set_enabled(false);

                static torch::Tensor hannWindow0 = createHannWindow(
                    {seq.img0.index({t}).sizes()[2], seq.img0.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow1 = createHannWindow(
                    {seq.img1.index({t}).sizes()[2], seq.img1.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow2 = createHannWindow(
                    {seq.img2.index({t}).sizes()[2], seq.img2.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow3 = createHannWindow(
                    {seq.img3.index({t}).sizes()[2], seq.img3.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow4 = createHannWindow(
                    {seq.img4.index({t}).sizes()[2], seq.img4.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow5 = createHannWindow(
                    {seq.img5.index({t}).sizes()[2], seq.img5.index({t}).sizes()[3]}, _device);
                static torch::Tensor hannWindow6 = createHannWindow(
                    {seq.img6.index({t}).sizes()[2], seq.img6.index({t}).sizes()[3]}, _device);

                // Spectrum loss
                torch::Tensor spectrumLoss = /*std::min(_lossLevel, 1.0) * */ _spectrumLossWeight * (
                    frameLossWeight0 * imageSpectrumLoss(
                        out.img0.to(torch::kFloat32) * hannWindow0,
                        seq.img0.index({t}).to(torch::kFloat32) * hannWindow0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 *
                        imageSpectrumLoss(
                        out.img1.to(torch::kFloat32) * hannWindow1,
                        seq.img1.index({t}).to(torch::kFloat32) * hannWindow1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 *
                        imageSpectrumLoss(
                        out.img2.to(torch::kFloat32) * hannWindow2,
                        seq.img2.index({t}).to(torch::kFloat32) * hannWindow2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 *
                        imageSpectrumLoss(
                        out.img3.to(torch::kFloat32) * hannWindow3,
                        seq.img3.index({t}).to(torch::kFloat32) * hannWindow3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 *
                        imageSpectrumLoss(
                        out.img4.to(torch::kFloat32) * hannWindow4,
                        seq.img4.index({t}).to(torch::kFloat32) * hannWindow4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 *
                        imageSpectrumLoss(
                        out.img5.to(torch::kFloat32) * hannWindow5,
                        seq.img5.index({t}).to(torch::kFloat32) * hannWindow5) : zero) +
                    (_lossLevel > 5.0 ? frameLossWeight6 *
                        imageSpectrumLoss(
                        out.img6.to(torch::kFloat32) * hannWindow6,
                        seq.img6.index({t}).to(torch::kFloat32) * hannWindow6) : zero)/* +
                    (_lossLevel > 6.0 ? frameLossWeight7 *
                        imageSpectrumLoss(
                        out.img7.to(torch::kFloat32),
                        seq.img7.index({t}).to(torch::kFloat32)) : zero)*/);
//                printf("spectrumLoss: %0.5f\n", spectrumLoss.item<double>());
                spectrumLossAcc += spectrumLoss.item<double>();

                // Total auxiliary losses
                torch::Tensor auxiliaryLosses =
                    encodingDistributionLoss.to(torch::kFloat32) +
                    encodingDistanceLoss.to(torch::kFloat32) +
                    encodingMeanLoss.to(torch::kFloat32) +
                    encodingCovarianceLoss.to(torch::kFloat32) +
                    encodingPrevDistanceLoss.to(torch::kFloat32) +
                    encodingDiscriminationLoss.to(torch::kFloat32) +
                    encodingCircularLoss.to(torch::kFloat32) +
                    encodingMaskLoss.to(torch::kFloat32) +
                    discriminationLoss.to(torch::kFloat32) +
                    frameClassificationLoss.to(torch::kFloat32) +
                    spectrumLoss.to(torch::kFloat32);
                auxiliaryLossesAcc += auxiliaryLosses.item<double>();

                // Frame decoding losses
#if 0
                torch::Tensor frameLoss = _frameLossWeight*(
                    frameLossWeight0 * yuvLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * yuvLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * yuvLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * yuvLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * yuvLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * yuvLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 5.0 ? frameLossWeight6 * yuvLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 6.0 ? frameLossWeight7 * yuvLoss(in.img7, out.img7) : zero));
                frameLossAcc += frameLoss.item<double>();
                torch::Tensor frameGradLoss = _frameGradLossWeight*(
                    frameLossWeight0 * imageGradLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageGradLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageGradLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageGradLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageGradLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageGradLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 5.0 ? frameLossWeight6 * imageGradLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 6.0 ? frameLossWeight7 * imageGradLoss(in.img7, out.img7) : zero));
                frameGradLossAcc += frameGradLoss.item<double>();
                torch::Tensor frameLaplacianLoss = _frameLaplacianLossWeight*(
                    frameLossWeight0 * imageLaplacianLoss(in.img0, out.img0) +
                    (_lossLevel > 0.0 ? frameLossWeight1 * imageLaplacianLoss(in.img1, out.img1) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 * imageLaplacianLoss(in.img2, out.img2) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 * imageLaplacianLoss(in.img3, out.img3) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 * imageLaplacianLoss(in.img4, out.img4) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 * imageLaplacianLoss(in.img5, out.img5) : zero) +
                    (_lossLevel > 5.0 ? frameLossWeight6 * imageLaplacianLoss(in.img6, out.img6) : zero) +
                    (_lossLevel > 6.0 ? frameLossWeight7 * imageLaplacianLoss(in.img7, out.img7) : zero));
                frameLaplacianLossAcc += frameLaplacianLoss.item<double>();
#endif

                // Frame reconstruction losses
//                torch::Tensor reconstructionLosses = frameLoss + frameGradLoss + frameLaplacianLoss;

                torch::Tensor reconstructionLosses = 1.0*(
                    frameLossWeight0 *
                        yuvImageLoss(out.img0.to(torch::kFloat32), seq.img0.index({t}).to(torch::kFloat32)) +
                    (_lossLevel > 0.0 ? frameLossWeight1 *
                        yuvImageLoss(out.img1.to(torch::kFloat32), seq.img1.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 1.0 ? frameLossWeight2 *
                        yuvImageLoss(out.img2.to(torch::kFloat32), seq.img2.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 2.0 ? frameLossWeight3 *
                        yuvImageLoss(out.img3.to(torch::kFloat32), seq.img3.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 3.0 ? frameLossWeight4 *
                        yuvImageLoss(out.img4.to(torch::kFloat32), seq.img4.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 4.0 ? frameLossWeight5 *
                        yuvImageLoss(out.img5.to(torch::kFloat32), seq.img5.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 5.0 ? frameLossWeight6 *
                        yuvImageLoss(out.img6.to(torch::kFloat32), seq.img6.index({t}).to(torch::kFloat32)) : zero) +
                    (_lossLevel > 6.0 ? frameLossWeight7 *
                        yuvImageLoss(out.img7.to(torch::kFloat32), seq.img7.index({t}).to(torch::kFloat32)) : zero));
                reconstructionLossesAcc += reconstructionLosses.item<double>();

                // Total loss
                torch::Tensor loss = reconstructionLosses + auxiliaryLosses;
                lossAcc += loss.item<double>();

                // Encoder-decoder backward pass
                loss.backward();

                at::autocast::set_enabled(true);

                // Display
                if (b == displayFrameId) {
                    int displaySeqId = rnd() % doot2::batchSize;

                    // Input images
                    MultiLevelImage inImage;
                    scaleDisplayImages(
                        MultiLevelImage {
                            seq.img0.index({t}).index({displaySeqId}), seq.img1.index({t}).index({displaySeqId}),
                            seq.img2.index({t}).index({displaySeqId}), seq.img3.index({t}).index({displaySeqId}),
                            seq.img4.index({t}).index({displaySeqId}), seq.img5.index({t}).index({displaySeqId}),
                            seq.img6.index({t}).index({displaySeqId}),
                            //in.img7.index({displaySeqId}), 0.0
                            torch::zeros({3, 480, 640}, TensorOptions().device(_device)),
                            0.0
                        },
                        inImage, torch::kCPU
                    );
//                    _trainingInfo->images["input7"].write()->copyFrom(inImage.img7.data_ptr<float>());
                    _trainingInfo->images["input6"].write()->copyFrom(inImage.img6.data_ptr<float>());
                    _trainingInfo->images["input5"].write()->copyFrom(inImage.img5.data_ptr<float>());
                    _trainingInfo->images["input4"].write()->copyFrom(inImage.img4.data_ptr<float>());
                    _trainingInfo->images["input3"].write()->copyFrom(inImage.img3.data_ptr<float>());
                    _trainingInfo->images["input2"].write()->copyFrom(inImage.img2.data_ptr<float>());
                    _trainingInfo->images["input1"].write()->copyFrom(inImage.img1.data_ptr<float>());
                    _trainingInfo->images["input0"].write()->copyFrom(inImage.img0.data_ptr<float>());

                    // TODO temp
                    at::autocast::set_enabled(false);
                    MultiLevelImage in {
                        seq.img0.index({(int)t}),
                        seq.img1.index({(int)t}),
                        seq.img2.index({(int)t}),
                        seq.img3.index({(int)t}),
                        seq.img4.index({(int)t}),
                        seq.img5.index({(int)t}),
                        seq.img6.index({(int)t}),
                        torch::zeros({16, 3, 480, 640}, TensorOptions().device(_device)),
                        //seq.img7.index({(int) t}),
                        _lossLevel
                    };
                    torch::Tensor* inRefs[] = {&in.img0, &in.img1, &in.img2, &in.img3, &in.img4, &in.img5, &in.img6, &in.img7};
                    torch::Tensor* outRefs[] = {&out.img0, &out.img1, &out.img2, &out.img3, &out.img4, &out.img5, &out.img6, &out.img7};
                    for (auto i=0; i<7; ++i) {
                        auto [mean, variance, uvCovar] = localVariance(
                            inRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        auto [varErrorY, varErrorUV, l2ErrorY, l2ErrorUV] = yuvImageErrors(
                            inRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32),
                            outRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        auto totalLoss = yuvImageCombinedError(
                            inRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32),
                            outRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        uvCovar = tf::interpolate((uvCovar * 200.0 + 0.5)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["uvCovar" + std::to_string(i)].write()->copyFrom(uvCovar.data_ptr<float>());
                        varErrorY = tf::interpolate((varErrorY * 1.0)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["varErrorY" + std::to_string(i)].write()->copyFrom(varErrorY.data_ptr<float>());
                        varErrorUV = tf::interpolate((varErrorUV * 0.5)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["varErrorUV" + std::to_string(i)].write()->copyFrom(varErrorUV.data_ptr<float>());
                        l2ErrorY = tf::interpolate((l2ErrorY * 50.0)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["l2ErrorY" + std::to_string(i)].write()->copyFrom(l2ErrorY.data_ptr<float>());
                        l2ErrorUV = tf::interpolate((l2ErrorUV * 15.0)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["l2ErrorUV" + std::to_string(i)].write()->copyFrom(l2ErrorUV.data_ptr<float>());
                        totalLoss = tf::interpolate(totalLoss
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 640}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["totalLoss" + std::to_string(i)].write()->copyFrom(totalLoss.data_ptr<float>());

                        auto [spectrumAmp, spectrumPhase] = imageSpectrum(
                            inRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        spectrumAmp = tf::interpolate((spectrumAmp.index({Slice(), Slice(0,1)}) * 0.1)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumAmp" + std::to_string(i)].write()->copyFrom(spectrumAmp.data_ptr<float>());
                        spectrumPhase = tf::interpolate((spectrumPhase.index({Slice(), Slice(0,1)}) * M_1_PI * 0.5 + 0.5)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumPhase" + std::to_string(i)].write()->copyFrom(spectrumPhase.data_ptr<float>());
                        auto [spectrumAmpPrediction, spectrumPhasePrediction] = imageSpectrum(
                            outRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        spectrumAmpPrediction = tf::interpolate((spectrumAmpPrediction.index({Slice(), Slice(0,1)}) * 0.1)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumAmpPrediction" + std::to_string(i)].write()->copyFrom(spectrumAmpPrediction.data_ptr<float>());
                        spectrumPhasePrediction = tf::interpolate((spectrumPhasePrediction.index({Slice(), Slice(0,1)}) * M_1_PI * 0.5 + 0.5)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumPhasePrediction" + std::to_string(i)].write()->copyFrom(spectrumPhasePrediction.data_ptr<float>());
                        auto [spectrumAmpError, spectrumPhaseError] = imageSpectrumErrors(
                            inRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32),
                            outRefs[i]->index({displaySeqId}).unsqueeze(0).to(torch::kFloat32));
                        spectrumAmpError = tf::interpolate((spectrumAmpError.index({Slice(), Slice(0,1)}) * 0.1)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumAmpError" + std::to_string(i)].write()->copyFrom(spectrumAmpError.data_ptr<float>());
                        spectrumPhaseError = tf::interpolate((spectrumPhaseError.index({Slice(), Slice(0,1)}) * M_1_PI)
                            .to(torch::kFloat32), tf::InterpolateFuncOptions()
                            .size(std::vector<long>{480, 320}).mode(kNearestExact).align_corners(false))
                            .permute({0, 2, 3, 1}).squeeze().contiguous().to(torch::kCPU);
                        _trainingInfo->images["spectrumPhaseError" + std::to_string(i)].write()->copyFrom(spectrumPhaseError.data_ptr<float>());
                    }
                    at::autocast::set_enabled(true);
                    // TODO end of temp

                    // Output images
                    MultiLevelImage outImage;
                    scaleDisplayImages(
                        MultiLevelImage {
                            out.img0.index({displaySeqId}), out.img1.index({displaySeqId}),
                            out.img2.index({displaySeqId}), out.img3.index({displaySeqId}),
                            out.img4.index({displaySeqId}), out.img5.index({displaySeqId}),
                            out.img6.index({displaySeqId}),
                            //out.img7.index({displaySeqId}),
                            torch::zeros({16, 3, 480, 640}, TensorOptions().device(_device)), 0.0
                        },
                        outImage, torch::kCPU
                    );
//                    _trainingInfo->images["prediction7"].write()->copyFrom(outImage.img7.data_ptr<float>());
                    _trainingInfo->images["prediction6"].write()->copyFrom(outImage.img6.data_ptr<float>());
                    _trainingInfo->images["prediction5"].write()->copyFrom(outImage.img5.data_ptr<float>());
                    _trainingInfo->images["prediction4"].write()->copyFrom(outImage.img4.data_ptr<float>());
                    _trainingInfo->images["prediction3"].write()->copyFrom(outImage.img3.data_ptr<float>());
                    _trainingInfo->images["prediction2"].write()->copyFrom(outImage.img2.data_ptr<float>());
                    _trainingInfo->images["prediction1"].write()->copyFrom(outImage.img1.data_ptr<float>());
                    _trainingInfo->images["prediction0"].write()->copyFrom(outImage.img0.data_ptr<float>());

                    // Output images decoded from random encodings
                    if (_useEncodingCircularLoss || _useDiscriminator) {
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

                    if (_useEncodingCovarianceLoss)
                        _trainingInfo->images["covariance_matrix"].write()->copyFrom(covarianceMatrixCPU.contiguous().data_ptr<float>());

                    torch::Tensor encodingImage;
                    encodingImage = enc.index({displaySeqId}).to(torch::kCPU, torch::kFloat32).reshape({32, 64})*0.05 + 0.5;
                    encodingImage = tf::interpolate(encodingImage.unsqueeze(0).unsqueeze(0),
                        tf::InterpolateFuncOptions()
                            .size(std::vector<long>{32*8, 64*8})
                            .mode(kNearestExact).align_corners(false)
                    );
                    _trainingInfo->images["encoding"].write()->copyFrom(encodingImage.contiguous().data_ptr<float>());

                    torch::Tensor encodingMaskImage;
                    encodingMaskImage = encMask.index({displaySeqId}).to(torch::kCPU, torch::kFloat32).reshape({32, 64});
                    encodingMaskImage = tf::interpolate(encodingMaskImage.unsqueeze(0).unsqueeze(0),
                        tf::InterpolateFuncOptions()
                            .size(std::vector<long>{32*8, 64*8})
                            .mode(kNearestExact).align_corners(false)
                    );
                    _trainingInfo->images["encoding_mask"].write()->copyFrom(encodingMaskImage.contiguous().data_ptr<float>());
                }

                c10::cuda::CUDACachingAllocator::emptyCache();
            }
#if 0
            // Encoding discriminator training pass
            if (_useEncodingDiscriminationLoss) {
                _encodingDiscriminator->zero_grad();
                _frameClassifier->zero_grad();
                for (int d = 0; d < _discriminatorVirtualBatchSize; ++d) {
                    int t = rnd() % sequenceLength;
                    torch::Tensor encodingDiscriminatorLoss;
                    torch::Tensor frameClassifierLoss;

                    // Encodings from the frame encoder
                    MultiLevelImage inTrue {
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
                    torch::Tensor encTrue = std::get<0>(_frameEncoder(inTrue));
                    MultiLevelImage inGenerated = _frameDecoder(torch::randn(
                        {doot2::batchSize, doot2::encodingLength},
                        TensorOptions().device(_device).dtype(torch::kBFloat16)), _lossLevel);
                    inGenerated.img0 = inGenerated.img0.detach();
                    inGenerated.img1 = inGenerated.img1.detach();
                    inGenerated.img2 = inGenerated.img2.detach();
                    inGenerated.img3 = inGenerated.img3.detach();
                    inGenerated.img4 = inGenerated.img4.detach();
                    inGenerated.img5 = inGenerated.img5.detach();
                    inGenerated.img6 = inGenerated.img6.detach();
                    inGenerated.img7 = inGenerated.img7.detach();
                    torch::Tensor encGenerated = std::get<0>(_frameEncoder(inGenerated));

                    // Train the frame classifier
                    if (_useFrameClassificationLoss) {
                        {
                            torch::Tensor classification = _frameClassifier(encTrue);
                            frameClassifierLoss = torch::binary_cross_entropy_with_logits(classification,
                                torch::ones_like(classification));
                        }
                        {
                            torch::Tensor classification = _frameClassifier(encGenerated);
                            frameClassifierLoss += torch::binary_cross_entropy_with_logits(classification,
                                torch::zeros_like(classification));
                        }
                        frameClassifierLossAcc += frameClassifierLoss.item<double>();
                        frameClassifierLoss.backward();
                    }

                    // Train the encoding discriminator
                    if (_useEncodingDiscriminationLoss) {
                        if (d%2 == 0) { // Encodings from images (true or generated) are both considered fake from
                            // the p.o.v. of the discriminator - alternate between them.
                            // Encodings from real images
                            torch::Tensor discrimination = _encodingDiscriminator(encTrue.detach());
                            encodingDiscriminatorLoss = torch::binary_cross_entropy_with_logits(discrimination,
                                torch::zeros_like(discrimination));
                        }
                        else {
                            // Encodings from fake images
                            torch::Tensor discrimination = _encodingDiscriminator(encGenerated.detach());
                            encodingDiscriminatorLoss = torch::binary_cross_entropy_with_logits(discrimination,
                                torch::zeros_like(discrimination));
                        }
                        {   // Gaussian prior encodings
                            encRandom = torch::randn({doot2::batchSize, doot2::encodingLength},
                                TensorOptions().device(_device).dtype(torch::kBFloat16));
                            torch::Tensor discrimination = _encodingDiscriminator(encRandom);
                            encodingDiscriminatorLoss += torch::binary_cross_entropy_with_logits(discrimination,
                                torch::ones_like(discrimination));
                        }
                        encodingDiscriminatorLossAcc += encodingDiscriminatorLoss.item<double>();
                        encodingDiscriminatorLoss.backward();
                    }

                    torch::Tensor rndEncodingImage;
                    rndEncodingImage = encRandom.index({0}).to(torch::kCPU, torch::kFloat32).reshape({32, 64})*0.15 + 0.5;
                    rndEncodingImage = tf::interpolate(rndEncodingImage.unsqueeze(0).unsqueeze(0),
                        tf::InterpolateFuncOptions()
                            .size(std::vector<long>{32*8, 64*8})
                            .mode(kNearestExact).align_corners(false)
                    );
                    _trainingInfo->images["random_encoding"].write()->copyFrom(rndEncodingImage.contiguous().data_ptr<float>());
                }
            }
#endif
            // Update training parameters according to scheduling
            updateTrainingParameters(nVirtualBatchesPerCycle);
            ++_trainingIteration;

            // Clip the gradients
            nn::utils::clip_grad_norm_(_frameEncoder->parameters(), 1.0, 2.0, true);
            nn::utils::clip_grad_norm_(_frameDecoder->parameters(), 1.0, 2.0, true);

            // Apply gradients
            _optimizer->step();
            if (_useDiscriminator)
                _discriminatorOptimizer->step();
            if (_useEncodingDiscriminationLoss)
                _encodingDiscriminatorOptimizer->step();
            _frameEncoder->zero_grad();
            _frameDecoder->zero_grad();
            if (_useDiscriminator) {
                _discriminator->zero_grad();
                _frameClassifier->zero_grad();
            }
            if (_useEncodingDiscriminationLoss)
                _encodingDiscriminator->zero_grad();
        }

        lossAcc /= framesPerCycle;
        frameLossAcc /= framesPerCycle;
        frameGradLossAcc /= framesPerCycle;
        frameLaplacianLossAcc /= framesPerCycle;
        frameClassificationLossAcc /= framesPerCycle;
        frameClassifierLossAcc /= (double)nVirtualBatchesPerCycle*(double)_discriminatorVirtualBatchSize;
        encodingDistributionLossAcc /= framesPerCycle;
        encodingDistanceLossAcc /= framesPerCycle;
        encodingMeanLossAcc /= framesPerCycle;
        encodingCovarianceLossAcc /= framesPerCycle;
        encodingPrevDistanceLossAcc /= framesPerCycle;
        encodingDiscriminationLossAcc /= framesPerCycle;
        encodingDiscriminatorLossAcc /= (double)nVirtualBatchesPerCycle*(double)_discriminatorVirtualBatchSize;
        discriminationLossAcc /= framesPerCycle;
        discriminatorLossAcc /= (double)nVirtualBatchesPerCycle*(double)_discriminatorVirtualBatchSize;
        encodingCircularLossAcc /= framesPerCycle;
        encodingMaskLossAcc /= framesPerCycle;
        spectrumLossAcc /= framesPerCycle;
        reconstructionLossesAcc /= framesPerCycle;
        auxiliaryLossesAcc /= framesPerCycle;

        // Update spectrum loss weight so that it tends to match with reconstruction losses
        if (_lossLevel > 0.0)
            _spectrumLossWeight *= _spectrumLossFiltering + (1.0-_spectrumLossFiltering)*
                (reconstructionLossesAcc / spectrumLossAcc);

        // Loss level adjustment
        double controlP, controlI, controlD;
        {
            constexpr double P = 0.005;
            constexpr double I = 0.0002;
            constexpr double D = 0.05;
            constexpr double errorFilter = 0.95;
            double error = _targetReconstructionLoss - reconstructionLossesAcc;
            if (error < 0.0) // damp the integrated error when lossLevel is larger than target
                _lossLevelErrorIntegrate *= 0.99;
            _lossLevelErrorIntegrate += error;
            double errorDeriv = -_lossLevelErrorFiltered;
            _lossLevelErrorFiltered = errorFilter * _lossLevelErrorFiltered + (1.0 - errorFilter) * error;
            errorDeriv += _lossLevelErrorFiltered;

            _lossLevel += _lossLevelErrorFiltered * P + _lossLevelErrorIntegrate * I + errorDeriv * D;

            _lossLevel = std::clamp(_lossLevel, 0.0, 7.0);
            if (_lossLevel < 1.0e-8) {
                _lossLevelErrorIntegrate = 0.0;
                _lossLevelErrorFiltered = 0.0;
            }

            controlP = _lossLevelErrorFiltered * P;
            controlI = _lossLevelErrorIntegrate * I;
            controlD = errorDeriv * D;
        }

        // Write the time series
        {
            auto timeSeriesWriteHandle = _trainingInfo->trainingTimeSeries.write();
            auto currentTime = high_resolution_clock::now();

            timeSeriesWriteHandle->addEntries(
                "time", (double)duration_cast<milliseconds>(currentTime-_trainingStartTime).count() / 1000.0,
                "frameLoss", frameLossAcc,
                "frameGradLoss", frameGradLossAcc,
                "frameLaplacianLoss", frameLaplacianLossAcc,
                "frameClassificationLoss", frameClassificationLossAcc,
                "frameClassifierLoss", frameClassifierLossAcc,
                "encodingDistributionLoss", encodingDistributionLossAcc,
                "encodingDistanceLoss", encodingDistanceLossAcc,
                "encodingMeanLoss", encodingMeanLossAcc,
                "encodingCovarianceLoss", encodingCovarianceLossAcc,
                "encodingPrevDistanceLoss", encodingPrevDistanceLossAcc,
                "encodingDiscriminationLoss", encodingDiscriminationLossAcc,
                "encodingDiscriminatorLoss", encodingDiscriminatorLossAcc,
                "discriminationLoss", discriminationLossAcc,
                "discriminatorLoss", discriminatorLossAcc,
                "encodingCircularLoss", encodingCircularLossAcc,
                "encodingMaskLoss", encodingMaskLossAcc,
                "spectrumLoss", spectrumLossAcc,
                "spectrumLossWeight", _spectrumLossWeight,
                "reconstructionLosses", reconstructionLossesAcc,
                "auxiliaryLosses", auxiliaryLossesAcc,
                "loss", lossAcc,
                "lossLevel", _lossLevel,
                "lossLevelErrorIntegrate", _lossLevelErrorIntegrate,
                "lossLevelErrorFiltered", _lossLevelErrorFiltered,
                "controlP", controlP,
                "controlI", controlI,
                "controlD", controlD,
                "optimizerLearningRate", _optimizerLearningRate,
                "optimizerBeta1", _optimizerBeta1,
                "optimizerWeightDecay", _optimizerWeightDecay,
                "maxEncodingCovariance", maxEncodingCovariance
            );
        }

        lossAcc = 0.0;
        frameLossAcc = 0.0;
        frameGradLossAcc = 0.0;
        frameLaplacianLossAcc = 0.0;
        frameClassificationLossAcc = 0.0;
        frameClassifierLossAcc = 0.0;
        encodingDistributionLossAcc = 0.0;
        encodingDistanceLossAcc = 0.0;
        encodingMeanLossAcc = 0.0;
        encodingCovarianceLossAcc = 0.0;
        encodingPrevDistanceLossAcc = 0.0;
        encodingDiscriminationLossAcc = 0.0;
        encodingDiscriminatorLossAcc = 0.0;
        discriminationLossAcc = 0.0;
        discriminatorLossAcc = 0.0;
        encodingCircularLossAcc = 0.0;
        encodingMaskLossAcc = 0.0;
        spectrumLossAcc = 0.0;
        reconstructionLossesAcc = 0.0;
        auxiliaryLossesAcc = 0.0;

        if (_abortTraining)
            break;
    }

    at::autocast::clear_cache();
    at::autocast::set_enabled(false);
}

MultiLevelImage MultiLevelAutoEncoderModel::scaleSequences(const SequenceStorage& storage)
{
    MultiLevelImage image;
    const auto* storageFrames7 = storage.getSequence<float>("frame");
    const auto* storageFrames6 = storage.getSequence<float>("frame6");
    const auto* storageFrames5 = storage.getSequence<float>("frame5");
    const auto* storageFrames4 = storage.getSequence<float>("frame4");
    const auto* storageFrames3 = storage.getSequence<float>("frame3");
    const auto* storageFrames2 = storage.getSequence<float>("frame2");
    const auto* storageFrames1 = storage.getSequence<float>("frame1");
    const auto* storageFrames0 = storage.getSequence<float>("frame0");

//    image.img7 = storageFrames7->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
//    assert(image.img7.sizes()[0] == storage.length());

    // Create scaled sequence data
    if (storageFrames6 != nullptr) {
        image.img6 = storageFrames6->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img6.sizes()[0] == storage.length());
    }
    else {
        image.img6 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 240, 320},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img6.index_put_({t}, tf::interpolate(image.img7.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{240, 320}).mode(kArea)));
    }
    if (storageFrames5 != nullptr) {
        image.img5 = storageFrames5->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img5.sizes()[0] == storage.length());
    }
    else {
        image.img5 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 120, 160},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img5.index_put_({t}, tf::interpolate(image.img6.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{120, 160}).mode(kArea)));
    }
    if (storageFrames4 != nullptr) {
        image.img4 = storageFrames4->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img4.sizes()[0] == storage.length());
    }
    else {
        image.img4 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 60, 80},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img4.index_put_({t}, tf::interpolate(image.img5.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{60, 80}).mode(kArea)));
    }
    if (storageFrames3 != nullptr) {
        image.img3 = storageFrames3->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img3.sizes()[0] == storage.length());
    }
    else {
        image.img3 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 30, 40},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img3.index_put_({t}, tf::interpolate(image.img4.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{30, 40}).mode(kArea)));
    }
    if (storageFrames2 != nullptr) {
        image.img2 = storageFrames2->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img2.sizes()[0] == storage.length());
    }
    else {
        image.img2 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 15, 20},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img2.index_put_({t}, tf::interpolate(image.img3.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{15, 20}).mode(kArea)));
    }
    if (storageFrames1 != nullptr) {
        image.img1 = storageFrames1->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img1.sizes()[0] == storage.length());
    }
    else {
        image.img1 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 15, 10},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
            image.img1.index_put_({t}, tf::interpolate(image.img2.index({t}), tf::InterpolateFuncOptions()
                .size(std::vector<long>{15, 10}).mode(kArea)));
    }
    if (storageFrames0 != nullptr) {
        image.img0 = storageFrames0->tensor().to(_device, torch::kBFloat16).permute({0, 1, 4, 2, 3}); // permute into TBCHW
        assert(image.img0.sizes()[0] == storage.length());
    }
    else {
        image.img0 = torch::zeros({(long)storage.length(), image.img7.sizes()[1], image.img7.sizes()[2], 5, 5},
            TensorOptions().device(_device).dtype(torch::kBFloat16));
        for (int t = 0; t < storage.length(); ++t)
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
    //image.img7 = orig.img7.permute({1, 2, 0}).contiguous().to(device, torch::kFloat32);
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

void MultiLevelAutoEncoderModel::updateTrainingParameters(int64_t nVirtualBatchesPerCycle)
{
    double iterationNormalized = (double)_trainingIteration / nVirtualBatchesPerCycle;
    double warmupFactor = 0.5-0.5*std::cos(std::min(iterationNormalized / _warmupDuration, 1.0)*M_PI);

    _optimizerLearningRate = _optimizerLearningRateInitial + warmupFactor*
        (_optimizerLearningRateFinal-_optimizerLearningRateInitial);
    _optimizerBeta1 = _optimizerBeta1Initial + warmupFactor*
        (_optimizerBeta1Final-_optimizerBeta1Initial);
    _optimizerWeightDecay = _optimizerWeightDecayInitial + warmupFactor*
        (_optimizerWeightDecayFinal-_optimizerWeightDecayInitial);

    for (auto& group : _optimizer->param_groups()) {
        dynamic_cast<torch::optim::AdamWOptions&>(group.options())
            .lr(_optimizerLearningRate)
            .betas({_optimizerBeta1, _optimizerBeta2})
            .weight_decay(_optimizerWeightDecay);
    }
}

#pragma once

#include <filesystem>


namespace doot2
{
constexpr uint32_t encodingLength = 2048;
constexpr uint32_t batchSize = 16;
constexpr uint32_t actionVectorLength = 6;
constexpr uint32_t sequenceLength = 64;
constexpr uint32_t frameWidth = 640;
constexpr uint32_t frameHeight = 480;
static const std::filesystem::path modelsDirectory {"models"};
static const std::filesystem::path frameEncoderFilename {modelsDirectory / "frame_encoder.pt"};
static const std::filesystem::path frameDecoderFilename {modelsDirectory / "frame_decoder.pt"};
static const std::filesystem::path flowDecoderFilename {modelsDirectory / "flow_decoder.pt"};
}

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

static const std::filesystem::path experimentsDirectory {"experiments/"};
static const std::filesystem::path guiLayoutFilename {"gui_layout.json"};
}

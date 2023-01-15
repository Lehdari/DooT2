#pragma once

namespace doot2
{
constexpr uint32_t encodingLength = 2048;
constexpr uint32_t batchSize = 16;
constexpr uint32_t actionVectorLength = 6;
constexpr uint32_t sequenceLength = 64;
constexpr uint32_t frameWidth = 640;
constexpr uint32_t frameHeight = 480;
constexpr char frameEncoderFilename[] {"models/frame_encoder.pt"};
constexpr char frameDecoderFilename[] {"models/frame_decoder.pt"};
constexpr char flowDecoderFilename[] {"models/flow_decoder.pt"};
}

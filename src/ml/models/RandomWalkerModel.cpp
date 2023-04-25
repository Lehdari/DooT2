//
// Project: DooT2
// File: RandomWalkerModel.cpp
//
// Copyright (c) 2023 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ml/models/RandomWalkerModel.hpp"
#include "Constants.hpp"
#include "Heatmap.hpp"

#include "gvizdoom/DoomGame.hpp"
#include <opencv2/highgui.hpp> // TODO temp


using namespace ml;
using namespace gvizdoom;


RandomWalkerModel::RandomWalkerModel(Heatmap* heatmap) :
    _heatmap    (heatmap)
{
    reset();
}

void RandomWalkerModel::reset()
{
    auto& doomGame = DoomGame::instance();

    _actionPrev = torch::zeros({doot2::actionVectorLength});
    _heatmapValueAbsMovingAverage = 1.0f;
    Vec2f playerPos = doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0);
    _playerPosPrev = playerPos;
    _playerVelocityMovingAverage = 1.0f;

    // Reset the heatmap
    if (_heatmap != nullptr) {
        _heatmap->reset(playerPos);
        _heatmap->applyExitPositionPriori(doomGame.getGameState<GameState::ExitPos>(), 0.25f);
        _heatmapValuePrev = _heatmap->sample(playerPos);
    }
}

void RandomWalkerModel::init(const nlohmann::json& experimentConfig)
{
}

void RandomWalkerModel::infer(const TensorVector& input, TensorVector& output)
{
    auto& doomGame = DoomGame::instance();

    // Compute heatmap value improvement
    float heatmapValue = _heatmap->sample(doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0));
    float heatmapValueDiff = heatmapValue - _heatmapValuePrev;
    _heatmapValuePrev = heatmapValue;
    // Normalize with the moving average
    float heatmapValueDiffNormalized = heatmapValueDiff / _heatmapValueAbsMovingAverage;
    _heatmapValueAbsMovingAverage = 0.99f*_heatmapValueAbsMovingAverage + 0.01f*std::abs(heatmapValueDiff);

    // Compute the player velocity
    Vec2f playerPos = doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0);
    float playerVelocity = (playerPos - _playerPosPrev).norm();
    _playerPosPrev = playerPos;
    // Normalize with the moving average
    float playerVelocityNormalized = playerVelocity / _playerVelocityMovingAverage;
    _playerVelocityMovingAverage = 0.99f*_playerVelocityMovingAverage + 0.01f*playerVelocity;

    // Some fuzzy logic to determine the amount of randomness to be introduced into the output:
    // if the heatmap value starts to grow (moving into areas that has been visited often)
    // or the player velocity starts to drop (stuck in a corner or something like that)
    // grow the amount of randomness to be introduced
    float randomnessFactor = 1.0f -
        std::clamp(1.0f-heatmapValueDiffNormalized, 0.0f, 1.0f)*
        std::clamp(playerVelocityNormalized, 0.0f, 1.0f);

    output.resize(1);
    torch::Tensor newAction = _actionPrev+randomnessFactor*torch::randn({doot2::actionVectorLength});
    output[0] = 0.95f*(0.75f*_actionPrev + 0.25f*newAction);
    output[0] = torch::clamp(output[0], -1.0f, 1.0f); // clamp it so the values don't grow ridiculously large
    _actionPrev = output[0];

    // Update heatmap
    _heatmap->addGaussianSample(
        doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0), 1.0f, 100.0f);
}

void RandomWalkerModel::trainImpl(SequenceStorage& storage)
{
}

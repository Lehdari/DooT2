//
// Project: DooT2
// File: HeatmapActionModule.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "HeatmapActionModule.hpp"

#include <gvizdoom/DoomGame.hpp>

#include <opencv2/highgui.hpp>


using namespace gvizdoom;


HeatmapActionModule::HeatmapActionModule(const HeatmapActionModule::Settings &settings) :
    _settings           (settings),
    _heatmap            (settings.resolution, settings.resolution, CV_32FC1, 0.0f),
    _heatmapMaxValue    (0.0f),
    _heatmapNormalized  (settings.resolution, settings.resolution, CV_32FC1, 0.0f)
{
}

void HeatmapActionModule::applyExitPositionPriori(Vec2f exitPos, float scale)
{
    // transform world coords to heatmap coords
    exitPos = exitPos/_settings.cellSize + Vec2f(_settings.resolution/2.0f, _settings.resolution/2.0f);
    for (int j=0; j<_settings.resolution; ++j) {
        auto* p = _heatmap.ptr<float>(j);
        for (int i=0; i<_settings.resolution; ++i) {
            p[i] = (Vec2f(i+0.5f, j+0.5f) - exitPos).norm() * scale;
            if (p[i] > _heatmapMaxValue)
                _heatmapMaxValue = p[i];
        }
    }

    refreshNormalization();
}

void HeatmapActionModule::addSample(const Vec2f& playerPos, float s)
{
    int heatmapMiddle = _settings.resolution / 2;
    Vec2f playerPosMap = playerPos / _settings.cellSize + Vec2f(heatmapMiddle, heatmapMiddle);

    int px = playerPosMap(0);
    int py = playerPosMap(1);
    Vec2f frac(playerPosMap(0)-px, playerPosMap(1)-py);
    addSubSample(px, py, s*(1.0f-frac(0))*(1.0f-frac(1)));
    addSubSample(px+1, py, s*frac(0)*(1.0f-frac(1)));
    addSubSample(px, py+1, s*(1.0f-frac(0))*frac(1));
    addSubSample(px+1, py+1, s*frac(0)*frac(1));

    float h = sample(playerPosMap);
    if (h > _heatmapMaxValue) {
        _heatmapMaxValue = h;
    }
}

void HeatmapActionModule::addGaussianSample(const Vec2f& playerPos, float s, float sigma)
{
    float nSigma = sigma / _settings.cellSize;
    int r = std::ceil(nSigma*3.0f);

    // generate gaussian distribution
    float scale = (1.0f / (nSigma*std::sqrt(2.0f*(float)M_PI)));
    static std::vector<float> gaussian(2*r+1, 0.0f);
    gaussian.resize(2*r+1);
    for (int i=0; i<gaussian.size(); ++i) {
        gaussian[i] = scale*std::exp(-0.5f*std::pow((i-r)/nSigma, 2.0f));
    }

    for (int j=-r; j<=r; ++j) {
        for (int i=-r; i<=r; ++i) {
            addSample(playerPos+_settings.cellSize*Vec2f(i,j), s*gaussian[i+r]*gaussian[j+r]);
        }
    }
}

void HeatmapActionModule::refreshNormalization()
{
    _heatmapNormalized = _heatmap / _heatmapMaxValue;
}

void HeatmapActionModule::reset()
{
    _heatmap = cv::Mat(_settings.resolution, _settings.resolution, CV_32FC1, 0.0f);
    _heatmapNormalized = cv::Mat(_settings.resolution, _settings.resolution, CV_32FC1, 0.0f);
    _heatmapMaxValue = 0.0f;
    state = State();
}

void HeatmapActionModule::operator()(
    const ActionManager::CallParams& callParams,
    ActionManager::UpdateParams& updateParams,
    ActionManager& actionManager
) {
    float heatmapSample = sample(callParams.playerPos, true);
    state.diff = heatmapSample-state.samplePrev;
    state.samplePrev = heatmapSample;

    auto& actionVector = actionManager.getActionVector();
    if (state.diff > 0.15f) {
        // invert action in case heatmap value grows rapidly (we're approaching region
        // that has been visited often)
        updateParams.actionVectorOverwrite.resize(actionVector.size());
        for (int i=0; i<actionVector.size(); ++i)
            updateParams.actionVectorOverwrite[i] = -1.0f*actionVector[i];
    }
    else if (state.diff < 0.0f) {
        // duplicate actions if heatmap value is decreasing
        updateParams.actionVectorOverwrite = actionVector;
    }
}

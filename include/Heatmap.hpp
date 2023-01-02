//
// Project: DooT2
// File: Heatmap.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ActionManager.hpp"
#include "MathTypes.hpp"

#include <opencv2/core/mat.hpp>


class Heatmap {
public:
    struct Settings {
        int     resolution; // width and height of the exploration heatmap
        float   cellSize;   // exploration heatmap cell size
    };

    Heatmap(const Settings& settings);

    void addSample(const Vec2f& playerPos, float s=1.0f);
    void addGaussianSample(const Vec2f& playerPos, float s, float sigma);
    void refreshNormalization();

    inline float sample(Vec2f p, bool worldCoords=false) const noexcept;
    inline float normalizedSample(Vec2f p, bool worldCoords=false) const noexcept;

    void reset();

    void operator()(
        const ActionManager::CallParams& callParams,
        ActionManager::UpdateParams& updateParams);

    // TODO temp
    float getDiff() const { return _diff; }

private:
    Settings    _settings;

    cv::Mat     _heatmap;
    float       _heatmapMaxValue;
    cv::Mat     _heatmapNormalized;

    // State for action synthesis
    float       _samplePrev;
    float       _diff;

    inline void addSubSample(int x, int y, float s);
};


float Heatmap::sample(Vec2f p, bool worldCoords) const noexcept
{
    if (worldCoords)
        p = p/_settings.cellSize + Vec2f(_settings.resolution/2.0f, _settings.resolution/2.0f);
    int px = p(0);
    int py = p(1);
    Vec2f frac(p(0)-px, p(1)-py);
    return (1.0f-frac(0))*(1.0f-frac(1))*_heatmap.at<float>(py, px) +
           frac(0)*(1.0f-frac(1))*_heatmap.at<float>(py, px+1) +
           (1.0f-frac(0))*frac(1)*_heatmap.at<float>(py+1, px) +
           frac(0)*frac(1)*_heatmap.at<float>(py+1, px+1);
}

float Heatmap::normalizedSample(Vec2f p, bool worldCoords) const noexcept
{
    if (worldCoords)
        p = p/_settings.cellSize + Vec2f(_settings.resolution/2.0f, _settings.resolution/2.0f);
    int px = p(0);
    int py = p(1);
    Vec2f frac(p(0)-px, p(1)-py);
    return (1.0f-frac(0))*(1.0f-frac(1))*_heatmapNormalized.at<float>(py, px) +
           frac(0)*(1.0f-frac(1))*_heatmapNormalized.at<float>(py, px+1) +
           (1.0f-frac(0))*frac(1)*_heatmapNormalized.at<float>(py+1, px) +
           frac(0)*frac(1)*_heatmapNormalized.at<float>(py+1, px+1);
}

void Heatmap::addSubSample(int x, int y, float s)
{
    _heatmap.at<float>(y, x) += s;
}

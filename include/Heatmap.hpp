//
// Project: DooT2
// File: HeatmapActionModule.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once

#include "ActionManager.hpp"
#include "util/MathTypes.hpp"

#include <opencv2/core/mat.hpp>


class Heatmap {
public:
    struct Settings {
        int     resolution;     // width and height of the exploration heatmap
        float   cellSize;       // exploration heatmap cell size
        Vec2f   playerInitPos;  // location where player starts the map (will be used as the heatmap center)
    };

    Heatmap(const Settings& settings);

    void applyExitPositionPriori(const Vec2f& exitPos, float scale=0.25f);

    void addSample(const Vec2f& playerPos, float s=1.0f);
    void addGaussianSample(const Vec2f& playerPos, float s, float sigma);
    void refreshNormalization();

    inline float sample(Vec2f p) const noexcept;
    inline float normalizedSample(Vec2f p) const noexcept;

    void reset(const Vec2f& playerInitPos);

private:
    Settings    _settings;

    cv::Mat     _heatmap;
    float       _heatmapMaxValue;
    cv::Mat     _heatmapNormalized;

    // Convert point in world coordinates to heatmap coordinates
    inline float sampleInternal(Vec2f p) const noexcept;
    inline float normalizedSampleInternal(Vec2f p) const noexcept;
    inline Vec2f toHeatmapCoords(const Vec2f& p) const;
    inline void addSubSample(int x, int y, float s);
};


float Heatmap::sample(Vec2f p) const noexcept
{
    return sampleInternal(toHeatmapCoords(p));
}

float Heatmap::normalizedSample(Vec2f p) const noexcept
{
    return normalizedSampleInternal(toHeatmapCoords(p));;
}

float Heatmap::sampleInternal(Vec2f p) const noexcept
{
    int px = p(0);
    int py = p(1);
    Vec2f frac(p(0)-px, p(1)-py);
    return (1.0f-frac(0))*(1.0f-frac(1))*_heatmap.at<float>(py, px) +
           frac(0)*(1.0f-frac(1))*_heatmap.at<float>(py, px+1) +
           (1.0f-frac(0))*frac(1)*_heatmap.at<float>(py+1, px) +
           frac(0)*frac(1)*_heatmap.at<float>(py+1, px+1);
}

float Heatmap::normalizedSampleInternal(Vec2f p) const noexcept
{
    int px = p(0);
    int py = p(1);
    Vec2f frac(p(0)-px, p(1)-py);
    return (1.0f-frac(0))*(1.0f-frac(1))*_heatmapNormalized.at<float>(py, px) +
           frac(0)*(1.0f-frac(1))*_heatmapNormalized.at<float>(py, px+1) +
           (1.0f-frac(0))*frac(1)*_heatmapNormalized.at<float>(py+1, px) +
           frac(0)*frac(1)*_heatmapNormalized.at<float>(py+1, px+1);
}

Vec2f Heatmap::toHeatmapCoords(const Vec2f& p) const
{
    return (p-_settings.playerInitPos)/_settings.cellSize +
        Vec2f(_settings.resolution/2.0f, _settings.resolution/2.0f);
}

void Heatmap::addSubSample(int x, int y, float s)
{
    _heatmap.at<float>(y, x) += s;
}

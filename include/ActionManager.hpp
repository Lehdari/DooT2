//
// Project: DooT2
// File: ActionManager.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include "ActionConverter.hpp"

#include <random>
#include <opencv2/core/mat.hpp>


class ActionManager {
public:
    struct Settings {
        int     heatmapSize;        // width and height of the exploration heatmap
        float   heatmapCellSize;    // exploration heatmap cell size
    };

    ActionManager(const Settings& settings);

    gvizdoom::Action operator()(float playerXRelative, float playerYRelative);

    float sampleHeatmap()

private:
    Settings                                _settings;
    ActionConverter<float>                  _actionConverter;
    cv::Mat                                 _heatmap;
    float                                   _heatmapMaxValue;
    cv::Mat                                 _heatmapNormalized;

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;

    gvizdoom::Action generateRandomAction();
};

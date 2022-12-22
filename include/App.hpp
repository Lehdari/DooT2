//
// Project: DooT2
// File: App.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#pragma once


#include <SDL.h>
#include <opencv2/core/mat.hpp>
#include "gvizdoom/Action.hpp"

#include "ActionConverter.hpp"

#include <random>


class App {
public:
    App();
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*             _window;
    SDL_Renderer*           _renderer;
    SDL_Texture*            _texture;

    bool                    _quit;
    ActionConverter<float>  _actionConverter;
    cv::Mat                 _positionPlot;

    static std::default_random_engine       _rnd;
    static std::normal_distribution<float>  _rndNormal;

    gvizdoom::Action generateRandomAction();
};

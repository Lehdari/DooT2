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


#include "ActionManager.hpp"
#include "HeatmapActionModule.hpp"
#include "DoorTraversalActionModule.hpp"
#include "SequenceStorage.hpp"
#include "ModelProto.hpp"

#include <SDL.h>
#include <opencv2/core/mat.hpp>
#include "gvizdoom/Action.hpp"

#include <random>


class App {
public:
    App();
    // TODO RO5
    ~App();

    void loop();

private:
    using Rnd = std::default_random_engine;

    Rnd                         _rnd;

    SDL_Window*                 _window;
    SDL_Renderer*               _renderer;
    SDL_Texture*                _texture;

    bool                        _quit;
    ActionManager               _actionManager;
    HeatmapActionModule         _heatmapActionModule;
    DoorTraversalActionModule   _doorTraversalActionModule;
    SequenceStorage             _sequenceStorage;
    cv::Mat                     _positionPlot;
    Vec2f                       _initPlayerPos;

    size_t                      _frameId;
    size_t                      _batchEntryId;
    bool                        _newPatchReady;

    ModelProto                  _model;

    torch::Device               _torchDevice;
    FrameEncoder                _frameEncoder;
    FrameDecoder                _frameDecoder;

    
    void nextMap(); // proceed to next map
};

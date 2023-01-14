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
#include "DoorTraversalActionModule.hpp"
#include "HeatmapActionModule.hpp"
#include "ModelProto.hpp"
#include "RewardModelTrainer.hpp"
#include "SequenceStorage.hpp"

#include <SDL.h>
#include <opencv2/core/mat.hpp>
#include "gvizdoom/Action.hpp"

#include <random>


class App {
public:
    App();
    App(const App&) = delete;
    App(App&&) = delete;
    App& operator=(const App&) = delete;
    App& operator=(App&&) = delete;

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

    ModelProto                  _modelEdec;
    RewardModelTrainer          _modelReward;

    torch::Device               _torchDevice;
    FrameEncoder                _frameEncoder;
    FrameDecoder                _frameDecoder;
    bool                        _trainRewardModel;
    
    void nextMap(); // proceed to next map
};

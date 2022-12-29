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

#include "ActionManager.hpp"
#include "Heatmap.hpp"
#include "SequenceStorage.hpp"
#include "ModelProto.hpp"


class App {
public:
    App();
    // TODO RO5
    ~App();

    void loop();

private:
    SDL_Window*     _window;
    SDL_Renderer*   _renderer;
    SDL_Texture*    _texture;

    bool            _quit;
    ActionManager   _actionManager;
    Heatmap         _heatmap;
    SequenceStorage _sequenceStorage;
    cv::Mat         _positionPlot;

    size_t          _frameId;
    size_t          _batchId;
    bool            _newPatchReady;

    ModelProto      _model;

    void nextMap(); // proceed to next map
};

//
// Project: DooT2
// File: App.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "App.hpp"

#include "gvizdoom/DoomGame.hpp"

#include <opencv2/highgui.hpp>


using namespace gvizdoom;


constexpr std::size_t batchSize = 16;


App::App() :
    _rnd                        (1507715517),
    _window                     (nullptr),
    _renderer                   (nullptr),
    _texture                    (nullptr),
    _quit                       (false),
    _heatmapActionModule        (HeatmapActionModule::Settings{256, 32.0f}),
    _doorTraversalActionModule  (),
    _sequenceStorage            (batchSize, 64),
    _positionPlot               (1024, 1024, CV_32FC3, cv::Scalar(0.0f)),
    _frameId                    (0),
    _batchId                    (0),
    _newPatchReady              (false)
{
    auto& doomGame = DoomGame::instance();

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Error: Could not initialize SDL!\n");
        return;
    }
    _window = SDL_CreateWindow(
        "DooT2",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        doomGame.getScreenWidth(),
        doomGame.getScreenHeight(),
        SDL_WINDOW_SHOWN);
    if (_window == nullptr) {
        printf("Error: SDL Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }
    _renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED);
    if (_renderer == nullptr) {
        printf("Error: SDL Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }
    _texture = SDL_CreateTexture(
        _renderer,
        SDL_PIXELFORMAT_BGRA32,
        SDL_TEXTUREACCESS_STREAMING,
        doomGame.getScreenWidth(),
        doomGame.getScreenHeight());
    if (_texture == nullptr) {
        printf("Error: SDL Texture could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }

    // Setup ActionManager
    _actionManager.addModule(&_heatmapActionModule);
    _actionManager.addModule(&_doorTraversalActionModule);
}

App::~App()
{
    // Destroy SDL objects and quit SDL subsystems
    if (_texture != nullptr)
        SDL_DestroyTexture(_texture);
    if (_renderer != nullptr)
        SDL_DestroyRenderer(_renderer);
    if (_window != nullptr)
        SDL_DestroyWindow(_window);
}

void App::loop()
{
    auto& doomGame = DoomGame::instance();
    SDL_Event event;

    Vec2f playerPosScreen(0.0f, 0.0f);

    size_t recordBeginFrameId = 768+_rnd()%512;
    size_t recordEndFrameId = recordBeginFrameId+64;

    while (!_quit) {
        while(SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
                (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) ||
                (event.type == SDL_KEYDOWN &&
                event.key.keysym.sym == SDLK_ESCAPE)) {
                _quit = true;
            }
        }

        static Vec2f playerPosRelative(0.0f, 0.0f);
        if (_frameId >= recordEndFrameId || doomGame.update(_actionManager(
            {_frameId, playerPosRelative}
            ))) {
            nextMap();
            recordBeginFrameId = 768+_rnd()%512;
            recordEndFrameId = recordBeginFrameId+64;
            continue;
        }

        if (_frameId >= recordBeginFrameId) {
            auto recordFrameId = _frameId - recordBeginFrameId;
            _sequenceStorage[recordFrameId][_batchId].bgraFrame =
                Image<uint8_t>(doomGame.getScreenWidth(), doomGame.getScreenHeight(),
                ImageFormat::BGRA, doomGame.getPixelsBGRA());
        }

        // Render screen
        auto screenHeight = doomGame.getScreenHeight();
        auto screenWidth = doomGame.getScreenWidth();
        uint8_t* sdlPixels;
        int pitch;
        SDL_LockTexture(_texture, nullptr, reinterpret_cast<void**>(&sdlPixels), &pitch);
        assert(pitch == screenWidth*4);
        memcpy(sdlPixels, doomGame.getPixelsBGRA(), sizeof(uint8_t)*screenWidth*screenHeight*4);
        SDL_UnlockTexture(_texture);
        SDL_RenderCopy(_renderer, _texture, nullptr, nullptr);
        SDL_RenderPresent(_renderer);

        static const float initPlayerX = doomGame.getGameState<GameState::PlayerPos>()(0);
        static const float initPlayerY = doomGame.getGameState<GameState::PlayerPos>()(1);
        playerPosRelative(0) = doomGame.getGameState<GameState::PlayerPos>()(0) - initPlayerX;
        playerPosRelative(1) = initPlayerY - doomGame.getGameState<GameState::PlayerPos>()(1); // invert y

        // Update heatmap
        _heatmapActionModule.addGaussianSample(playerPosRelative, 1.0f, 100.0f);
        _heatmapActionModule.refreshNormalization();

        // Update position plot
        playerPosScreen = playerPosRelative * 0.125f;
        if (playerPosScreen(0) >= -512.0f && playerPosScreen(1) >= -512.0f &&
            playerPosScreen(0) < 512.0f && playerPosScreen(1) < 512.0f) {
            if (_heatmapActionModule.getDiff() > 0.0f)
                _positionPlot.at<Vec3f>((int)playerPosScreen(1)+512, (int)playerPosScreen(0)+512)(1) = 1.0f;
            else
                _positionPlot.at<Vec3f>((int)playerPosScreen(1)+512, (int)playerPosScreen(0)+512)(0) = 1.0f;
        }

        // Render position plot
        static uint64_t a = 0;
        if (++a%10 == 0) {
            // Heatmap to red channel
            for (int j=0; j<1023; ++j) {
                auto* p = _positionPlot.ptr<Vec3f>(j);
                for (int i=0; i<1023; ++i) {
                    p[i](0) *= 0.995f;
                    p[i](1) *= 0.995f;
                    p[i](2) = _heatmapActionModule.normalizedSample(Vec2f(i * 0.25f, j * 0.25f), false);
                }
            }

            cv::imshow("Position Plot", _positionPlot);
            cv::waitKey(1);
        }

        // Train
        if (_newPatchReady) {
            // Create copy of the sequence storage
            auto sequenceStorageCopy(_sequenceStorage);

            printf("Training...\n");
            _model.waitForTrainingFinish();
            _model.trainAsync(sequenceStorageCopy);
            _newPatchReady = false;
        }

        ++_frameId;
    }
}

void App::nextMap()
{
    auto& doomGame = DoomGame::instance();

    _actionManager.reset();
    _positionPlot *= 0.0f;

    _frameId = 0;
    if (++_batchId >= batchSize) {
        _newPatchReady = true;
        _batchId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    newGameConfig.map = _batchId + 1;

    doomGame.restart(newGameConfig);
}

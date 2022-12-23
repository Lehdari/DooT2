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


App::App() :
    _window             (nullptr),
    _renderer           (nullptr),
    _texture            (nullptr),
    _quit               (false),
    _heatmap            (Heatmap::Settings{256, 32.0f}),
    _positionPlot       (1024, 1024, CV_32FC3, cv::Scalar(0.0f))
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

        _heatmap.addSample(playerPosRelative);

        if (doomGame.update(_actionManager()))
            doomGame.restart();

        auto screenHeight = doomGame.getScreenHeight();
        auto screenWidth = doomGame.getScreenWidth();

        // Render screen
        uint8_t* sdlPixels;
        int pitch;
        SDL_LockTexture(_texture, nullptr, reinterpret_cast<void**>(&sdlPixels), &pitch);
        assert(pitch == screenWidth*4);
        memcpy(sdlPixels, doomGame.getPixelsBGRA(), sizeof(uint8_t)*screenWidth*screenHeight*4);
        SDL_UnlockTexture(_texture);
        SDL_RenderCopy(_renderer, _texture, nullptr, nullptr);
        SDL_RenderPresent(_renderer);

        static const float initPlayerX = doomGame.getGameState<GameState::X>();
        static const float initPlayerY = doomGame.getGameState<GameState::Y>();
        playerPosRelative(0) = doomGame.getGameState<GameState::X>() - initPlayerX;
        playerPosRelative(1) = initPlayerY - doomGame.getGameState<GameState::Y>(); // invert y
        printf("%0.5f %0.5f \n", playerPosRelative(0), playerPosRelative(1));
        Vec2f playerPosScreen = playerPosRelative * 0.125f;

        if (playerPosScreen(0) >= -512.0f && playerPosScreen(1) >= -512.0f &&
            playerPosScreen(0) < 512.0f && playerPosScreen(1) < 512.0f)
            _positionPlot.at<Vec3f>((int)playerPosScreen(1) + 512, (int)playerPosScreen(0) + 512)(1) = 1.0f;

        // Render position plot
        static uint64_t a = 0;
        if (++a%10 == 0) {
            for (int j=0; j<1023; ++j) {
                auto* p = _positionPlot.ptr<Vec3f>(j);
                for (int i=0; i<1023; ++i) {
                    p[i](1) *= 0.995f;
                    p[i](2) = _heatmap.normalizedSample(Vec2f(i*0.25f, j*0.25f));
                }
            }

            cv::imshow("Position Plot", _positionPlot);
            cv::waitKey(1);
        }
    }
}

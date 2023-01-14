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
#include "Constants.hpp"
#include <filesystem>
#include "gvizdoom/DoomGame.hpp"
#include <opencv2/highgui.hpp>

#include "Constants.hpp"

<<<<<<< HEAD
using namespace doot2;
using namespace gvizdoom;
=======

using namespace doot2;
using namespace gvizdoom;
using namespace torch;
using namespace torch::indexing;
namespace fs = std::filesystem;

>>>>>>> add encodings and heatmap-based rewards to sequence storage

App::App() :
    _rnd                        (1507715517),
    _window                     (nullptr),
    _renderer                   (nullptr),
    _texture                    (nullptr),
    _quit                       (false),
    _heatmapActionModule        (HeatmapActionModule::Settings{256, 32.0f}),
    _doorTraversalActionModule  (false),
    _sequenceStorage            (SequenceStorage::Settings{batchSize, sequenceLength, false, true, 0, 0, ImageFormat::BGRA, encodingLength}),
    _positionPlot               (1024, 1024, CV_32FC3, cv::Scalar(0.0f)),
    _initPlayerPos              (0.0f, 0.0f),
    _frameId                    (0),
    _batchEntryId               (0),
    _newPatchReady              (false),
    _torchDevice                (torch::cuda::is_available() ? kCUDA : kCPU)
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
    _actionManager.addModule(&_doorTraversalActionModule);
    _actionManager.addModule(&_heatmapActionModule);

    // Load frame encoder
    if (fs::exists(frameEncoderFilename)) {
        printf("App: Loading frame encoder model from %s\n", frameEncoderFilename); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameEncoderFilename);
        _frameEncoder->load(inputArchive);
        // Use the inference mode
        _frameEncoder->eval();
    }
    else {
        printf("No %s found. Initializing a new frame encoder model.\n", frameEncoderFilename); // TODO logging
    }

    _frameEncoder->to(_torchDevice);

    // Load frame decoder
    if (fs::exists(frameDecoderFilename)) {
        printf("App: Loading frame encoder model from %s\n", frameDecoderFilename); // TODO logging
        serialize::InputArchive inputArchive;
        inputArchive.load_from(frameDecoderFilename);
        _frameDecoder->load(inputArchive);
        // Use the inference mode
        _frameDecoder->eval();
    }
    else {
        printf("No %s found. Initializing a new frame encoder model.\n", frameDecoderFilename); // TODO logging
    }

    _frameDecoder->to(_torchDevice);


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

    // BHWC
    torch::Tensor pixelBuffer{torch::zeros({1, 480, 640, 4})};

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

        // Player position relative to the starting position (_initPlayerPos)
        static Vec2f playerPosRelative(0.0f, 0.0f);

        // Action for this timestep
        auto action = _actionManager(
            {_frameId, playerPosRelative}
        );

        // Update the game state, restart if required
        if (_frameId >= recordEndFrameId || doomGame.update(action)) {
            nextMap();
            recordBeginFrameId = 768+_rnd()%512;
            recordEndFrameId = recordBeginFrameId+64;
            continue;
        }

        // Player position is undefined before the first update, so _initPlayerPos has to be
        // defined here. Also apply the exit position priori to the heatmap action module.
        if (_frameId == 0) {
            _initPlayerPos = doomGame.getGameState<GameState::PlayerPos>().block<2,1>(0,0);
            Vec2f exitPos = doomGame.getGameState<GameState::ExitPos>() - _initPlayerPos;
            exitPos(1) *= -1.0f;

            _heatmapActionModule.applyExitPositionPriori(exitPos);
        }

        // Record
        if (_frameId >= recordBeginFrameId) {
            auto recordFrameId = _frameId - recordBeginFrameId;
            auto batch = _sequenceStorage[recordFrameId];
            batch.actions[_batchEntryId] = action;

            // Convert the game frame from uint8 to float
            const auto imageFormat{ImageFormat::BGRA};
            Image<uint8_t> frameUint8(doomGame.getScreenWidth(), doomGame.getScreenHeight(), imageFormat);
            Image<float> frameFloat(doomGame.getScreenWidth(), doomGame.getScreenHeight(), imageFormat);
            frameUint8.copyFrom(doomGame.getPixelsBGRA());
            convertImage(frameUint8, frameFloat);

            // Copy the float frame to a torch::Tensor
            const auto nPixels = doomGame.getScreenWidth() * doomGame.getScreenHeight() * getImageFormatNChannels(imageFormat);
            copyToTensor(frameFloat.data(), nPixels, pixelBuffer);

            // upload to GPU and permute to BCHW
            torch::Tensor pixelBufferGpu = pixelBuffer.to(_torchDevice);            
            pixelBufferGpu = pixelBufferGpu.permute({0,3,1,2});
            
            // encode
            torch::Tensor encoding = _frameEncoder(pixelBufferGpu);

            // Check sanity with decoder
            torch::Tensor decoding = _frameDecoder(encoding);
            decoding = decoding.permute({0,2,3,1}).contiguous();

            cv::Mat decodingOpencv(480, 640, CV_32FC4);
            copyFromTensor(decoding.to(torch::kCPU), (float*)decodingOpencv.ptr<float>(0), 640*480*4);

            cv::imshow("app-decoding", decodingOpencv);

            // store encoding to the sequence storage
            copyFromTensor(encoding.to(torch::kCPU), batch.encodings[_batchEntryId], encodingLength);

            // Update relative player position
            playerPosRelative(0) = doomGame.getGameState<GameState::PlayerPos>()(0) - _initPlayerPos(0);
            playerPosRelative(1) = _initPlayerPos(1) - doomGame.getGameState<GameState::PlayerPos>()(1); // invert y

            // Reward is negative heatmap value
            batch.rewards[_batchEntryId] = -_heatmapActionModule.sample(playerPosRelative, true);
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

        // Update relative player position
        playerPosRelative(0) = doomGame.getGameState<GameState::PlayerPos>()(0) - _initPlayerPos(0);
        playerPosRelative(1) = _initPlayerPos(1) - doomGame.getGameState<GameState::PlayerPos>()(1); // invert y

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
#if 0
        if (_newPatchReady) {
            // Create copy of the sequence storage
            auto sequenceStorageCopy(_sequenceStorage);

            printf("Training...\n");
            _model.waitForTrainingFinish();
            _model.trainAsync(std::move(sequenceStorageCopy));
            _newPatchReady = false;
        }
#endif
        ++_frameId;
    }
}

void App::nextMap()
{
    auto& doomGame = DoomGame::instance();

    _actionManager.reset();
    _positionPlot *= 0.0f;

    _frameId = 0;
    if (++_batchEntryId >= batchSize) {
        _newPatchReady = true;
        _batchEntryId = 0;
    }

    gvizdoom::GameConfig newGameConfig = doomGame.getGameConfig();
    newGameConfig.map = _batchEntryId + 1;

    doomGame.restart(newGameConfig);
}

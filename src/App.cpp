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
#include "ml/Trainer.hpp"
#include "ml/Model.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "glad/glad.h"

#include <chrono>
#include <filesystem>


using namespace ml;
using namespace gvizdoom;
using namespace doot2;
namespace fs = std::filesystem;


App::App(Trainer* trainer, Model* model) :
    _window     (nullptr),
    _glContext  (nullptr),
    _quit       (false),
    _trainer    (trainer),
    _model      (model)
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
        1920, // TODO settings
        1080, // TODO settings
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
    if (_window == nullptr) {
        printf("Error: SDL Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true); // TODO settings
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);

    _glContext = SDL_GL_CreateContext(_window);
    if (_glContext == nullptr) {
        printf("Error: SDL OpenGL context could not be created! SDL_Error: %s\n",
            SDL_GetError());
        return;
    }

    // Load OpenGL extensions
    if (!gladLoadGL()) {
        printf("Error: gladLoadGL failed\n");
        return;
    }

    // Initialize OpenGL
    glViewport(0, 0, 1920, 1080); // TODO settings
    glClearColor(0.2f, 0.2f, 0.2f, 1.f);
    glEnable(GL_DEPTH_TEST);

    // Initialize gui
    _gui.init(_window, &_glContext);
    _gui.update(_model);
    if (fs::exists(guiLayoutFilename))
        _gui.loadLayout(guiLayoutFilename);
    else
        _gui.createDefaultLayout();
}

App::~App()
{
    // Save the GUI layout
    _gui.saveLayout(guiLayoutFilename);

    // Destroy window and quit SDL subsystems
    if (_glContext != nullptr)
        SDL_GL_DeleteContext(_glContext);
    if (_window != nullptr)
        SDL_DestroyWindow(_window);
    SDL_Quit();
}

void App::loop()
{
    using namespace std::chrono;

    constexpr double framerate = 60.0; // TODO settings

    SDL_Event event;
    while (!_quit) {
        auto frameBegin = high_resolution_clock::now();

        while(SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT ||
                (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) ||
                (event.type == SDL_KEYDOWN &&
                event.key.keysym.sym == SDLK_ESCAPE)) {
                _quit = true;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        _gui.render(_window, _trainer, _model);

        // Introduce delay to cap the framerate
        auto frameEnd = high_resolution_clock::now();
        auto frameTime = duration_cast<microseconds>(frameEnd - frameBegin).count();
        int delayMs = std::max((int)std::floor((1000.0/framerate)-((double)frameTime*0.001)), 0);
        SDL_Delay(delayMs);

        // Swap draw and display buffers
        SDL_GL_SwapWindow(_window);
    }
}

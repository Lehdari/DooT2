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
#include "Trainer.hpp"
#include "Model.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "glad/glad.h"


using namespace doot2;
using namespace gvizdoom;


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
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
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

    // Initialize imgui
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForOpenGL(_window, _glContext);
    ImGui_ImplOpenGL3_Init("#version 460");

    // Initialize OpenGL
    glViewport(0, 0, 1920, 1080); // TODO settings
    glClearColor(0.2f, 0.2f, 0.2f, 1.f);
    glEnable(GL_DEPTH_TEST);
}

App::~App()
{
    // Destroy imgui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    // Destroy window and quit SDL subsystems
    if (_glContext != nullptr)
        SDL_GL_DeleteContext(_glContext);
    if (_window != nullptr)
        SDL_DestroyWindow(_window);
    SDL_Quit();
}

void App::loop()
{
    SDL_Event event;
    while (!_quit) {
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

        gui();

        // Swap draw and display buffers
        SDL_GL_SwapWindow(_window);
        SDL_Delay(10);
    }
}

void App::gui() const
{
    imGuiNewFrame();

    ImGui::Begin("Training");
    {
        auto stateReadHandle = _model->trainingState.read();
        double loss = stateReadHandle->contains("loss") ? stateReadHandle->at("loss").get<double>() : 0.0;
        ImGui::Text("Loss: %0.5f", loss);
    }
    ImGui::End();

    imGuiRender();
}

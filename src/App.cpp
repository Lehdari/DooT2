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
#include "ml/Models.hpp"
#include "ml/ModelTypeUtils.hpp"
#include "ml/Trainer.hpp"

#include "gvizdoom/DoomGame.hpp"
#include "glad/glad.h"
#include "util/ExperimentUtils.hpp"

#include <chrono>
#include <filesystem>


using namespace ml;
using namespace gvizdoom;
using namespace doot2;
namespace fs = std::filesystem;


App::App(Trainer* trainer) :
    _window         (nullptr),
    _glContext      (nullptr),
    _quit           (false),
    _trainer        (trainer),
    _gridSearchId   (0)
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
    _gui.setCallback("resetExperiment", [&](const gui::State& guiState){ resetExperiment(); });
    _gui.update(_trainer->getTrainingInfo());

    // Read gui layout from the layout file
    if (fs::exists(guiLayoutFilename))
        _gui.loadLayout(guiLayoutFilename, _window);
    else
        _gui.createDefaultLayout();

    resetExperiment();
}

App::~App()
{
    // Save the GUI layout
    _gui.saveLayout(guiLayoutFilename, _window);

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

        // Render gui
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        _gui.render(_window, _trainer);

        // Start / pause / stop training
        trainingControl();

        // Introduce delay to cap the framerate
        auto frameEnd = high_resolution_clock::now();
        auto frameTime = duration_cast<microseconds>(frameEnd - frameBegin).count();
        int delayMs = std::max((int)std::floor((1000.0/framerate)-((double)frameTime*0.001)), 0);
        SDL_Delay(delayMs);

        // Swap draw and display buffers
        SDL_GL_SwapWindow(_window);
    }

    // Quit trainer thread in case it's running
    if (_trainerThread.joinable()) {
        _trainer->quit();
        _trainerThread.join();
        _trainer->saveExperiment();
    }
}

void App::trainingControl()
{
    switch (_gui.getState().trainingStatus) {
        case gui::State::TrainingStatus::STOPPED: {
            if (_trainerThread.joinable()) {
                _trainer->quit();
                _trainerThread.join();
                _trainer->saveExperiment();
            }
        }   break;
        case gui::State::TrainingStatus::ONGOING: {
            if (!_trainer->isFinished()) // training running, continue business as usual
                break;

            // Trainer finished, clean up the experiment
            if (_trainerThread.joinable()) {
                _trainerThread.join();
                _trainer->saveExperiment();
                if (_gridSearchId >= _gridSearchParameters.size()) {
                    _gridSearchId = 0;
                    _gridSearchParameters.clear();
                    // TODO change the training status to stopped
                }
            }

            // grid search specified, create the flattened parameter list in case it
            if (_gui.getState().gridSearch && _gridSearchParameters.empty()) {
                if (_gui.getState().gridSearchModelConfigParams.empty()) {
                    printf("ERROR: Grid search specified but no model parameters specified.\n"); // TODO logging
                }
                else {
                    _gridSearchParameters = flattenGridSearchParameters(_gui.getState().gridSearchModelConfigParams);
                    _gridSearchId = 0;
                }
            }

            // no trainer thread running, launch it
            updateExperimentConfig(*_trainer->getExperimentConfig());
            _trainer->setupExperiment(); // needed for updated training info
            _gui.update(_trainer->getTrainingInfo()); // communicate potential changes in training info to gui
            _trainerThread = std::thread(&ml::Trainer::loop, _trainer);
        }   break;
        case gui::State::TrainingStatus::PAUSED: {
            // TODO, requires pausing interface for Model
        }   break;
    }
}

void App::updateExperimentConfig(nlohmann::json& experimentConfig)
{
    // TODO This is here so that when using macros in the experiment name a new directory is generated.
    // TODO It is planned be overridable from the experiment configuration GUI section.
    if (experimentConfig.contains("experiment_root"))
        experimentConfig.erase("experiment_root");

    experimentConfig["experiment_name"] = _gui.getState().experimentName;
    experimentConfig["pwad_filenames"] = { // TODO kovakoodattua paskaa
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_01.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_02.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_03.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_04.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_05.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_06.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_07.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_08.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_09.wad",
        assetsDir/"wads"/"micro_nomonsters"/"micro_nomonsters_10.wad"
    };

    if (!_gui.getState().experimentBase.empty())
        experimentConfig["experiment_base_root"] = experimentRootFromString(_gui.getState().experimentBase);
    experimentConfig["software_version"] = GIT_VERSION;

    // Apply grid search parameter changes
    if (_gui.getState().gridSearch && !_gridSearchParameters.empty()) {
        experimentConfig["model_config"].update(_gridSearchParameters[_gridSearchId]);
        experimentConfig["n_training_epochs"] = 256;

        // TODO logging
        printf("INFO: Starting grid search experiment %lu / %lu, params:\n",
            _gridSearchId+1, _gridSearchParameters.size());
        for (auto& [param, value] : _gridSearchParameters[_gridSearchId].items()) {
            std::cout << "    " << param << ": " << value << std::endl;
        }

        ++_gridSearchId;
    }
}

void App::resetExperiment()
{
    if (_gui.getState().trainingStatus != gui::State::TrainingStatus::STOPPED) {
        printf("WARNING: App::resetExperiment called when experiment is ongoing, omitting reset...\n"); // TODO logging
        return;
    }

    // Setup experiment config
    nlohmann::json experimentConfig;
    updateExperimentConfig(experimentConfig);
    experimentConfig["model_type"] = _gui.getState().modelTypeName;
    if (_gui.getState().baseExperimentConfig.contains("model_config")) {
        experimentConfig["model_config"] = _gui.getState().baseExperimentConfig["model_config"];
        // model config may change, store also the original
        experimentConfig["base_model_config"] = _gui.getState().baseExperimentConfig["model_config"];
    }
    else {
        modelTypeNameCallback(_gui.getState().modelTypeName, [&]<typename T_Model>() { // load the default model config
            experimentConfig["model_config"] = getDefaultModelConfig<T_Model>();
        });
    }

    _trainer->configureExperiment(std::move(experimentConfig));
    _gui.update(_trainer->getTrainingInfo());
}

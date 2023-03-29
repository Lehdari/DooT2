#include "App.hpp"
#include "Constants.hpp"
#include "Heatmap.hpp"
#include "ml/Trainer.hpp"
#include "util/Utils.hpp"
#include "ml/models/AutoEncoderModel.hpp"
#include "ml/models/RandomWalkerModel.hpp"

#include "CLI/CLI.hpp"
#include "gvizdoom/DoomGame.hpp"

#include <thread>

int main()
{
    CLI::App cliApp{"DooT2 Machine Learning Research Platform"};
    uint32_t cliBatchSize{doot2::batchSize};
    uint32_t cliSequenceLength{doot2::sequenceLength}; 

    cliApp.add_option("--batchsize,-b", cliBatchSize, "Batch size for the autoencoder model");
    cliApp.add_option("--sequencelen,-s", cliSequenceLength, "Sequence length for the autoencoder model");

    CLI11_PARSE(cliApp);

    // Initialize the game
    gvizdoom::GameConfig config{0, nullptr, false, true,
        640, 480, true,
        gvizdoom::GameConfig::HUD_DISABLED, 2, false,
        3, 1, 1
    };
    config.pwadFileNames = {{assetsDir/"wads"/"oblige01.wad"}};
    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);
    doomGame.update(gvizdoom::Action());

    // Initialize models
    ml::AutoEncoderModel model; // model to be trained
    Heatmap randomWalkerHeatmap({512, 32.0f,
        doomGame.getGameState<gvizdoom::GameState::PlayerPos>().block<2,1>(0,0)});
    ml::RandomWalkerModel agentModel(&randomWalkerHeatmap); // model used for agent

    ml::Trainer trainer(&model, &agentModel, nullptr, cliBatchSize, cliSequenceLength);
    App app(&trainer, &model);

    std::thread trainerThread(&ml::Trainer::loop, &trainer);
    app.loop();
    trainer.quit();
    trainerThread.join();
    return 0;
}
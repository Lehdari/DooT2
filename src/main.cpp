#include "App.hpp"
#include "Constants.hpp"
#include "CLI/CLI.hpp"
#include "Trainer.hpp"
#include "Utils.hpp"
#include "AutoEncoderModel2.hpp"

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

    gvizdoom::GameConfig config{0, nullptr, false, true,
        640, 480, true,
        gvizdoom::GameConfig::HUD_DISABLED, 2, false,
        3, 1, 1
    };
    config.pwadFileNames = {{assetsDir/"wads"/"oblige01.wad"}};

    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);


    AutoEncoderModel2 model;

    Trainer trainer(&model, cliBatchSize, cliSequenceLength);
    App app(&trainer, &model);

    std::thread trainerThread(&Trainer::loop, &trainer);
    app.loop();
    trainer.quit();
    trainerThread.join();
    return 0;
}
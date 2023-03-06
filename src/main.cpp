#include "App.hpp"
#include "CLI/CLI.hpp"
#include "Trainer.hpp"
#include "Utils.hpp"
#include "AutoEncoderModel.hpp"

#include "gvizdoom/DoomGame.hpp"

#include <thread>

int main()
{
    CLI::App cliApp{"DooT2 Machine Learning Research Platform"};
    uint32_t cliBatchSize; 
    cliApp.add_option("--batchsize,-b", cliBatchSize, "Batch size for the autoencoder model");

    CLI11_PARSE(cliApp);

    gvizdoom::GameConfig config{0, nullptr, false, true,
        640, 480, true,
        gvizdoom::GameConfig::HUD_DISABLED, 2, false,
        3, 1, 1
    };
    config.pwadFileNames = {{assetsDir/"wads"/"oblige01.wad"}};

    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);


    AutoEncoderModel model;

    Trainer trainer(&model);
    App app(&trainer, &model);

    std::thread trainerThread(&Trainer::loop, &trainer);
    app.loop();
    trainer.quit();
    trainerThread.join();
    return 0;
}
#include "gvizdoom/DoomGame.hpp"

#include "App.hpp"
#include "Utils.hpp"


int main()
{
    gvizdoom::GameConfig config{0, nullptr, false, true,
        640, 480, true,
        gvizdoom::GameConfig::HUD_DISABLED, 2, false,
        3, 1, 1
    };
    config.pwadFileNames = {{assetsDir/"wads"/"oblige01.wad"}};

    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);

    App app;
    app.loop();

    return 0;
}
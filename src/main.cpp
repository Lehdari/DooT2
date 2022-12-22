#include "gvizdoom/DoomGame.hpp"

#include "App.hpp"
#include "Utils.hpp"


int main()
{
    gvizdoom::GameConfig config;
    config.pwadFileNames = {{assetsDir/"wads"/"oblige01.wad"}};

    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);

    App app;
    app.loop();

    return 0;
}
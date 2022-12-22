#include "gvizdoom/DoomGame.hpp"

#include "App.hpp"


int main()
{
    gvizdoom::GameConfig config;

    auto& doomGame = gvizdoom::DoomGame::instance();
    doomGame.init(config);

    App app;
    app.loop();

    return 0;
}
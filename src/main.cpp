#include "DoomGame.hpp"


int main()
{
    gvizdoom::GameConfig config;

    auto& game = gvizdoom::DoomGame::instance();
    game.init(config);

    for (int i=0; i<1000; ++i) {
        game.update(gvizdoom::Action());
        printf("%d\r", i);
    }

    return 0;
}
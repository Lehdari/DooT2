DooT 2
======

DooT 2 is a learning project with the ultimate goal of teaching a deep learning agent
to play a video game from pixel input only using a single prosumer-grade GPU. The agent
should be capable of generalizing for previously unseen environments (within the same game)
from a very sparse reward input (ideally level completion / game end only).
The chosen methodology is abstract in a sense that it is applicable to any environment
conforming to the standard reinforcement learning observation - action - state update loop.

The original Doom was chosen as the environment due to various desirable properties:
- Very long sequences: 35 tics * 60 s * 10 min = 21000 frames for a medium length level,
  necessitating observation compression and compact state representation
- Partial state observation: in FPS games the player can only see what's directly
  in front of them in addition to various player status indicators such as health and ammo
- Very long cause-and-effect delays (for example picking a keycard will grant access to
  an area much later) requiring long-term memory and capability of making connections
  between events separated by significant amount of timesteps
- The game can be run effectively on CPU, allowing GPU to be fully utilized for training the
  ML model
- Pre-existing tools for procedurally generating infinite number of levels of varying difficulty

![screenshot](https://github.com/Lehdari/DooT2/blob/master/docs/doot2_2024-03-26_13-14-01.png?raw=true)

---

Setup
-----
Tested with CMake 3.25.1, gcc, g++ 10.3.0 and 11.3.0

1) Install dependencies (Ubuntu Linux 22.04):
```
$ sudo apt install libsdl2-dev libeigen3-dev libopencv-dev
```

2) Clone the repository, checkout the `develop` branch for the latest changes:
```
$ git clone git@github.com:Lehdari/DooT2.git
$ cd DooT2/
$ git checkout develop
```

3) Clone the submodules:
```
$ git submodule update --init --recursive
```

4) Build (note that PyTorch will be built when invoking CMake the first time, this will take
   quite a while depending on your hardware):
```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
$ ninja -j0 doot2
```

5) Run
```
$ ./doot2
```

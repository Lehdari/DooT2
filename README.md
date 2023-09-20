DooT 2
======

Bot for Doom, now with C++, blackjack and hookers!

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

4) Build (note that PyTorch will be built when invoking CMake):
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

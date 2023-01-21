DooT 2
======

Bot for Doom, now with C++, blackjack and hookers!

---

Setup
-----

- Download submodules:
```
git submodule update --init --recursive
```

- Install [libtorch](https://github.com/pytorch/pytorch)
  - Either build from source or install from [here](https://pytorch.org/get-started/locally/)
  - Build from source:
    - Install its dependencies
    - Tested with CMake 3.25.1, gcc, g++ 10.3.0 and 11.3.0
    - Run `setup.py` in the repo
    - Run `tools/build_libtorch.py`
  - Use prebuilt packages:
    - Choose LibTorch as Package
    - C++ as Language
    - Choose Compute platform accordingly
    - Download the zip, unzip to where ever you want
  - Regardless of your installation style use `-D LIBTORCH_DIR` to set the directory where you built/installed libtorch
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

- Build [libtorch](https://github.com/pytorch/pytorch) from source using your preferred compiler
  - Install its dependencies
  - Tested with CMake 3.25.1, gcc, g++ 10.3.0 and 11.3.0
  - Run `setup.py` in the repo
  - Run `tools/build_libtorch.py`
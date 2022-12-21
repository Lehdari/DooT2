cmake_minimum_required(VERSION 3.20)
project(DooT2)


# Add external dependencies
add_subdirectory(ext)


# Add the main executable
add_executable(doot2 src/main.cpp)
target_link_libraries(doot2
    PUBLIC  gvizdoom
)
set_target_properties(doot2 PROPERTIES CXX_STANDARD 20)

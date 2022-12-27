cmake_minimum_required(VERSION 3.20)
project(DooT2)


# Add external dependencies
add_subdirectory(ext)

find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)


# Source files
set(DOOT2_SOURCES
    src/main.cpp
    src/ActionManager.cpp
    src/App.cpp
    src/Heatmap.cpp
    src/SequenceStorage.cpp
)


# Add the main executable
add_executable(doot2 ${DOOT2_SOURCES})
target_include_directories(doot2
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${SDL2_INCLUDE_DIRS}
)
target_link_libraries(doot2
    PUBLIC  gvizdoom
    PUBLIC  ${SDL2_LIBRARIES}
    PUBLIC  Eigen3::Eigen
    PUBLIC  ${OpenCV_LIBS}
)
target_compile_definitions(doot2
    PUBLIC  ASSETS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"
)
set_target_properties(doot2 PROPERTIES
    CXX_STANDARD    20
)

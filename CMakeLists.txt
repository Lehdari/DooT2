cmake_minimum_required(VERSION 3.20)
project(DooT2)


set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "Supported CUDA architectures")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE PATH "CUDA Compiler location")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8 CACHE PATH "CUDA Toolkit location")


# Add external dependencies
add_subdirectory(ext)

find_package(SDL2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

option(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pytorch/torch")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${LIBTORCH_DIR})
set(Caffe2_DIR "${LIBTORCH_DIR}/share/cmake/Caffe2/") # ???
include_directories("${LIBTORCH_DIR}/lib/include") # ???
find_package(Torch REQUIRED)

# Source files
set(DOOT2_SOURCES
    src/main.cpp
    src/ActionManager.cpp
    src/App.cpp
    src/DoorTraversalActionModule.cpp
    src/FlowDecoder.cpp
    src/FrameDecoder.cpp
    src/FrameEncoder.cpp
    src/HeatmapActionModule.cpp
    src/ModelProto.cpp
    src/ResNeXtModule.cpp
    src/SequenceStorage.cpp
    src/TimeSeries.cpp
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
    PUBLIC  opencv_highgui
    PUBLIC  torch
    PUBLIC  nlohmann_json
)
target_compile_definitions(doot2
    PUBLIC  ASSETS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"
)
set_target_properties(doot2 PROPERTIES
    CXX_STANDARD    20
)


# Add tests
set(DOOT2_TEST_SOURCES
    src/TimeSeries.cpp
    src/tests/TestTimeSeries.cpp
)

include(GoogleTest)
add_executable(doot2_tests ${DOOT2_TEST_SOURCES})
target_include_directories(doot2_tests
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(doot2_tests PROPERTIES
    CXX_STANDARD    20
)
target_link_libraries(doot2_tests
    PUBLIC  GTest::gtest_main
    PUBLIC  gtest
    PUBLIC  pthread
    PUBLIC  nlohmann_json
)
gtest_add_tests(TARGET doot2_tests
    TEST_SUFFIX .noArgs
    TEST_LIST   noArgsTests
)
gtest_discover_tests(doot2_tests)

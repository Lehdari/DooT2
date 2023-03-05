cmake_minimum_required(VERSION 3.26)
project(DooT2)


# Add external dependencies
add_subdirectory(ext)

find_package(SDL2 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV REQUIRED)

# Source files
set(DOOT2_SOURCES
    src/main.cpp
    src/ActionManager.cpp
    src/App.cpp
    src/AutoEncoderModel.cpp
    src/DoorTraversalActionModule.cpp
    src/FlowDecoder.cpp
    src/FrameDecoder.cpp
    src/FrameEncoder.cpp
    src/GuiImageRelay.cpp
    src/HeatmapActionModule.cpp
    src/Model.cpp
    src/ResNeXtModule.cpp
    src/SequenceStorage.cpp
    src/TimeSeries.cpp
    src/Trainer.cpp
)


# Add the main executable
add_executable(doot2 ${DOOT2_SOURCES})
target_include_directories(doot2
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${SDL2_INCLUDE_DIRS}
    PUBLIC  ext/CLI11/include
)
target_link_libraries(doot2
    PUBLIC  gvizdoom
    PUBLIC  gut_opengl
    PUBLIC  ImPlot
    PUBLIC  ${SDL2_LIBRARIES}
    PUBLIC  Eigen3::Eigen
    PUBLIC  opencv_highgui
    PUBLIC  torch_interface
    PUBLIC  nlohmann_json
    PUBLIC  pthread
)
target_compile_definitions(doot2
    PUBLIC  ASSETS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"
)
set_target_properties(doot2 PROPERTIES
    CXX_STANDARD        20
    CUDA_ARCHITECTURES  native
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
    CXX_STANDARD        20
    CUDA_ARCHITECTURES  native
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

cmake_minimum_required(VERSION 3.26)
project(DooT2)

# Custom CMake modules
include(cmake/GitVersion.cmake)

# Fetch the software version hash from git
git_version()

# Add external dependencies
add_subdirectory(ext)

find_package(SDL2 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV REQUIRED)

# Source files
set(DOOT2_GUI_SOURCES
    src/gui/GameWindow.cpp
    src/gui/Gui.cpp
    src/gui/ImageRelay.cpp
    src/gui/ImagesWindow.cpp
    src/gui/PlotWindow.cpp
    src/gui/TrainingWindow.cpp
    src/gui/Window.cpp
)
set(DOOT2_ML_SOURCES
    src/ml/Model.cpp
    src/ml/Trainer.cpp
    src/ml/models/AutoEncoderModel.cpp
    src/ml/models/EncoderModel.cpp
    src/ml/models/MultiLevelAutoEncoderModel.cpp
    src/ml/models/RandomWalkerModel.cpp
    src/ml/modules/Attention.cpp
    src/ml/modules/Attention2D.cpp
    src/ml/modules/Discriminator.cpp
    src/ml/modules/EncodingDiscriminator.cpp
    src/ml/modules/FlowDecoder.cpp
    src/ml/modules/FrameDecoder.cpp
    src/ml/modules/FrameEncoder.cpp
    src/ml/modules/LinearResidualModule.cpp
    src/ml/modules/MultiLevelDecoderModule.cpp
    src/ml/modules/MultiLevelEncoderModule.cpp
    src/ml/modules/MultiLevelFrameDecoder.cpp
    src/ml/modules/MultiLevelFrameEncoder.cpp
    src/ml/modules/ResNetConvBlock.cpp
    src/ml/modules/ResNetDownscaleConvBlock.cpp
    src/ml/modules/ResNetLinearBlock.cpp
    src/ml/modules/ResNeXtModule.cpp
    src/ml/modules/SqueezeExcitation.cpp
    src/ml/modules/TransformerBlock.cpp
    src/ml/modules/ViTBlock.cpp
)
set(DOOT2_UTIL_SOURCES
    src/util/ExperimentUtils.cpp
    src/util/Image.cpp
    src/util/SequenceStorage.cpp
    src/util/TensorUtils.cpp
    src/util/TimeSeries.cpp
)
set(DOOT2_SOURCES
    ${DOOT2_GUI_SOURCES}
    ${DOOT2_ML_SOURCES}
    ${DOOT2_UTIL_SOURCES}
    src/main.cpp
    src/ActionManager.cpp
    src/App.cpp
    src/DoorTraversalActionModule.cpp
    src/Heatmap.cpp
)


# Add the main executable
add_executable(doot2 ${DOOT2_SOURCES})
target_include_directories(doot2
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${SDL2_INCLUDE_DIRS}
    PUBLIC  ext/CLI11/include
    PUBLIC  ext/GraphicsUtils/ext/stb
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
    PUBLIC  GIT_VERSION=\"${GIT_VERSION}\"
)
set_target_properties(doot2 PROPERTIES
    CXX_STANDARD        20
    CUDA_ARCHITECTURES  native
)


# Add tests
set(DOOT2_TEST_SOURCES
    src/util/TimeSeries.cpp
    src/util/SequenceStorage.cpp
    src/tests/TestTimeSeries.cpp
    src/tests/TestSequence.cpp
    src/tests/TestSequenceStorage.cpp
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
    PUBLIC  torch_interface
)
target_compile_definitions(doot2_tests
    PUBLIC  ASSETS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/assets\"
)
gtest_add_tests(TARGET doot2_tests
    TEST_SUFFIX .noArgs
    TEST_LIST   noArgsTests
)
gtest_discover_tests(doot2_tests)

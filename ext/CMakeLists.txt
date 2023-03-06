# Direct CMake submodules
add_subdirectory(GraphicsUtils)
add_subdirectory(GViZDoom)

# ImPlot
add_library(ImPlot SHARED
    implot/implot.cpp
    implot/implot_demo.cpp
    implot/implot_items.cpp
)
target_include_directories(ImPlot
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/implot
)
target_link_libraries(ImPlot
    PRIVATE gut_opengl
)

# PyTorch requires a bit of extra work:

# Create build directory
set(PYTORCH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/pytorch)
file(MAKE_DIRECTORY ${PYTORCH_BUILD_DIR})

# Directories for find_package
set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch)
set(Caffe2_DIR "${LIBTORCH_DIR}/share/cmake/Caffe2/")
set(Torch_DIR "${PYTORCH_BUILD_DIR}/build/")
set(ENV{TORCH_INSTALL_PREFIX} "${PYTORCH_BUILD_DIR}/build/")

# Build libtorch in case it has not been built yet
find_package(Torch QUIET)
if (TORCH_FOUND)
    message(STATUS "libtorch found, skipping the build")
else()
    message(STATUS "Building libtorch")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND             ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytorch/tools/build_libtorch.py --rerun-cmake
        WORKING_DIRECTORY   ${PYTORCH_BUILD_DIR}
        RESULT_VARIABLE     BUILD_LIBTORCH_RETURN
    )
    if (NOT BUILD_LIBTORCH_RETURN EQUAL "0")
        message(FATAL_ERROR "Unable to build libtorch")
    endif()
endif()

# At this point libtorch should be built
if (NOT TORCH_FOUND)
    set(Caffe2_DIR "${LIBTORCH_DIR}/share/cmake/Caffe2/")
    set(Torch_DIR "${PYTORCH_BUILD_DIR}/build/")
    set(ENV{TORCH_INSTALL_PREFIX} "${PYTORCH_BUILD_DIR}/build/")
    find_package(Torch REQUIRED)
endif()

# find_package sets the TORCH_INCLUDE_DIRS to point to build directory, correct this
set(TORCH_INCLUDE_DIRS
    ${LIBTORCH_DIR}/include
    ${LIBTORCH_DIR}/include/torch/csrc/api/include
)
set_target_properties(torch PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES   "${TORCH_INCLUDE_DIRS}"
)

# Add interface library to enable easy usage on the main CMakeLists (no need to hassle with
# the include directories and such)
add_library(torch_interface INTERFACE)
target_link_libraries(torch_interface
    INTERFACE   torch
)
target_include_directories(torch_interface
    INTERFACE   ${TORCH_INCLUDE_DIRS}
)
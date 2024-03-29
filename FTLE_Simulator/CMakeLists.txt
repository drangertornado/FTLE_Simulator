cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif ()

project(FTLE_Simulator VERSION 1.0 LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)  # Choose the desired C++ standard (e.g., 11, 14, 17, etc.)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA package
find_package(CUDA REQUIRED)
# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})
# Add CUDA cusolver library
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})

# Set additional module paths
# set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/external/optix")
# # Search for the OptiX libraries and include files
# find_package(OptiX REQUIRED)
# # Add Optix include directories
# include_directories(${OptiX_INCLUDE})

# Set CUDA architecture (optional)
# Set to compute capability 6.1 (Pascal architecture)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas -O3 -arch=sm_61")
# Set CUDA-specific flags to make constexpr functions callable and to make GLM work
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr --use_fast_math -diag-suppress 20012 -diag-suppress 20014")

if(MSVC)
    # Add /NODEFAULTLIB:LIBCMT to exclude LIBCMT library
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    # Set C++ flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
endif()

# Add executable target
add_executable(main
    main.cpp
    cuda/CUDACheck.h
    cuda/CUDABuffer.h
    cuda/CuSolver.h
    structures/AABB.h
    structures/Camera.h
    structures/Light.h
    structures/Material.h
    structures/Ray.h
    structures/Point.h
    structures/TransferFunction.h
    classes/Grid.h
    classes/Grid.cpp
    classes/Renderer.h
    classes/Renderer.cpp
    classes/Window.h
    classes/Window.cpp
    classes/Shader.h
    classes/Settings.h
    kernels/FTLECompute.cuh
    kernels/FTLECompute.cu
    kernels/VolumeRender.cuh
    kernels/VolumeRender.cu
)

# Include directories containing source header files
include_directories(cuda)
include_directories(structures)
include_directories(classes)
include_directories(kernels)

# Set CUDA architectures (example: compute capability 6.1 for Pascal)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61)

# Build GLM
set(GLM_DIR ${PROJECT_SOURCE_DIR}/external/glm)
include_directories(${GLM_DIR})
include_directories(${GLM_DIR}/ext)
add_subdirectory(${GLM_DIR} EXCLUDE_FROM_ALL)

# Build GLFW
set(GLFW_DIR ${PROJECT_SOURCE_DIR}/external/glfw)
include_directories(${GLFW_DIR}/include)
add_subdirectory(${GLFW_DIR} EXCLUDE_FROM_ALL)

# Build Glad
set(GLAD_DIR ${PROJECT_SOURCE_DIR}/external/glad)
include_directories(${GLAD_DIR}/include)
add_subdirectory(${GLAD_DIR} EXCLUDE_FROM_ALL)

# Link external libraries
target_link_libraries(main
    ${CUDA_LIBRARIES}   # Link CUDA libraries
    glm::glm            # Link glm library
    glfw                # Link glfw library
    glad                # Link glad library
)

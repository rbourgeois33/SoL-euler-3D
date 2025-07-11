cmake_minimum_required(VERSION 3.18)
project(MyThrustProject LANGUAGES CXX CUDA)

# Enable C++17 and CUDA features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)

# Use Thrust from CUDA toolkit (recommended)
find_package(CUDAToolkit REQUIRED)

# Add external library
add_subdirectory(${CMAKE_SOURCE_DIR}/external/inih)

# Add executable target
add_executable(SoL_Euler_3D src/main.cu)

# Include directories
target_include_directories(SoL_Euler_3D PRIVATE
    external/inih)

# Optional CUDA compile options
target_compile_options(SoL_Euler_3D PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Link external libraries
target_link_libraries(SoL_Euler_3D PRIVATE inih)
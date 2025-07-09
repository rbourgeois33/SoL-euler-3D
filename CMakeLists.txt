cmake_minimum_required(VERSION 3.28)
project(SoL-euler-3D LANGUAGES CXX CUDA)

# Set C++17 and CUDA architectures
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require CUDA architectures to be specified
if (NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES not set.\n"
        "Please specify it manually, e.g., via:\n"
        "  cmake -DCMAKE_CUDA_ARCHITECTURES=70 .."
    )
endif()


add_subdirectory(${CMAKE_SOURCE_DIR}/external/inih)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/nvbench)

# === Main CUDA executable ===
add_executable(SoL-euler-3D src/main.cu)

target_include_directories(SoL-euler-3D PRIVATE
    external/inih
    external/nvbench
)

target_link_libraries(SoL-euler-3D PRIVATE inih nvbench)
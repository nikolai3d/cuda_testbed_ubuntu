cmake_minimum_required(VERSION 3.12)
project(testbed_cpp20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")

include_directories(${CUDA_INCLUDE_DIRS})

# Set the output directory for the build executables
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/Debug)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/Release)
endif()


add_executable(cuda_info source/main.cpp)
target_link_libraries(cuda_info ${CUDA_LIBRARIES})

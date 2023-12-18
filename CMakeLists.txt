cmake_minimum_required(VERSION 3.12)
project(testbed_cpp20)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if the operating system is Windows
if(WIN32)
  # Set the CUDA toolkit directory for Windows
  set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
else()
  # Set the CUDA toolkit directory for Unix-like systems (Linux)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
  set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
endif()


set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")

include_directories(${CUDA_INCLUDE_DIRS})

# Set the output directory for the build executables
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/Debug)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/Release)
endif()

# Add the component NVML as a static library
add_library(nvml STATIC source/components/ia_nvml.cpp source/components/ia_nvml.hpp source/components/ia_nvml_context.h source/components/ia_nvml_context.cpp)

# Link the component to the main executable
add_executable(cuda_info source/main.cpp)
# target_link_libraries(cuda_info ${CUDA_LIBRARIES})
target_link_libraries(cuda_info nvml)
cmake_minimum_required(VERSION 3.12)
project(testbed_cpp20)

set(CMAKE_CXX_STANDARD 23)
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

# Add the component NVML as a static library
add_library(nvml STATIC source/components/ia_nvml.cpp source/components/ia_nvml.hpp source/components/ia_nvml_context.h source/components/ia_nvml_context.cpp)

# Link the component to the main executable
add_executable(cuda_info source/main.cpp)
target_link_libraries(cuda_info ${CUDA_LIBRARIES})
target_link_libraries(cuda_info nvml)

# Catch2 Unit Tests
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)



# Add an executable for your unit tests
add_executable(unit_tests source/tests/test_main.cpp) # Add your test source files here
target_link_libraries(unit_tests PRIVATE Catch2::Catch2)
target_include_directories(unit_tests PRIVATE Catch2::Catch2)
target_link_libraries(unit_tests PRIVATE nvml) # Link with your static library


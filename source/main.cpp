// #include <iostream>

// int main() {
//     std::cout << "Hello, World!" << std::endl;
//     return 0;
// }

#include <iostream>
#include <cuda_runtime.h>
#include "components/nvml.h"

int main() {
    ia_nvml::query_nvml_pointers();
    
    NVML_RT_CALL(pfn_nvmlInit_v2())

    console::couti << "NVML Initialized";
  
    std::uint32_t device_count = 0;

    NVML_RT_CALL(pfn_nvmlDeviceGetCount_v2(&device_count))

    console::couti << "NVML Device count: " << device_count;

    ML_EXPECTS(device_count > 0);

    nvmlDevice_t device = nullptr;

    console::couti << "NVML Acquiring Device PTR: ";

    NVML_RT_CALL(pfn_nvmlDeviceGetHandleByIndex_v2(0, &device))

    ML_EXPECTS(device!=nullptr);

    std::vector<char> device_name;
    device_name.resize(1024, '\0');
    
    NVML_RT_CALL(pfn_nvmlDeviceGetName(device, device_name.data(), 1023))

    console::couti << "NVML device name: `" << std::string(device_name.data()) << "`";

    NVML_RT_CALL(pfn_nvmlShutdown())

    console::couti << "NVML Shut down";


    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    }

    return 0;
}

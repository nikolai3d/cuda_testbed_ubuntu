// #include <iostream>

// int main() {
//     std::cout << "Hello, World!" << std::endl;
//     return 0;
// }

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "components/ia_nvml.h"
class MyStream {
 public:
  // Template overload of the << operator for any type.
  template <typename T>
  MyStream& operator<<(const T& value) {
    std::cout << value << std::endl;
    return *this;
  }
};

namespace console {
MyStream couti;
}

#define ML_EXPECTS(x)                                                                                                   \
  if (!(x)) {                                                                                                           \
    std::cerr << "ML_EXPECTS failed: " << #x << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
    std::exit(1);                                                                                                       \
  }

int main() {
  ia_nvml::NVMLFunctionTable::instance().initialize_nvml_function_pointers();

  IA_NVML_CALL(pfn_nvmlInit_v2())

  console::couti << "NVML Initialized";

  std::uint32_t device_count = 0;

  IA_NVML_CALL(pfn_nvmlDeviceGetCount_v2(&device_count))

  console::couti << "NVML Device count: " << device_count;

  ML_EXPECTS(device_count > 0);

  nvmlDevice_t device = nullptr;

  console::couti << "NVML Acquiring Device PTR: ";

  IA_NVML_CALL(pfn_nvmlDeviceGetHandleByIndex_v2(0, &device))

  ML_EXPECTS(device != nullptr);

  std::vector<char> device_name;
  device_name.resize(1024, '\0');

  IA_NVML_CALL(pfn_nvmlDeviceGetName(device, device_name.data(), 1023))

  console::couti << "NVML device name: `" << std::string(device_name.data()) << "`";

  IA_NVML_CALL(pfn_nvmlShutdown())

  console::couti << "NVML Shut down";

  int         deviceCount = 0;
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

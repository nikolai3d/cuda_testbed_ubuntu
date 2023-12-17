// #include <iostream>

// int main() {
//     std::cout << "Hello, World!" << std::endl;
//     return 0;
// }

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "components/ia_nvml.hpp"
#include "components/ia_nvml_context.h"
class MyStream {
 public:
  class Helper {
   public:
    Helper(std::ostream& stream) : _stream(stream) {
    }

    ~Helper() {
      _stream << _ss.str() << std::endl;
    }

    template <typename T>
    Helper& operator<<(const T& value) {
      _ss << value;
      return *this;
    }

    Helper(const Helper&) = delete;
    Helper& operator=(const Helper&) = delete;
    Helper(Helper&&) = default;
    Helper& operator=(Helper&&) = default;

   private:
    std::ostream&     _stream;
    std::stringstream _ss;
  };

  Helper operator<<(const std::ostream& (*pf)(std::ostream&)) {
    return Helper(std::cout);
  }

  template <typename T>
  Helper operator<<(const T& value) {
    Helper helper(std::cout);
    helper << value;
    return helper;
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

  ia_nvml::NVMLContext nvml_context;

#define NVML_CALL(NVML_FUNCTION, ...) IA_NVML_CALL(nvml_context._nvml_function_table, NVML_FUNCTION, __VA_ARGS__)

  console::couti << "NVML Initialized";

  std::uint32_t device_count = 0;

  NVML_CALL(nvmlDeviceGetCount_v2, &device_count);

  console::couti << "NVML Device count: " << device_count;

  ML_EXPECTS(device_count > 0);

  nvmlDevice_t device = nullptr;

  console::couti << "NVML Acquiring Device PTR: ";

  NVML_CALL(nvmlDeviceGetHandleByIndex_v2, 0, &device);

  ML_EXPECTS(device != nullptr);

  std::vector<char> device_name;
  device_name.resize(1024, '\0');

  NVML_CALL(nvmlDeviceGetName, device, device_name.data(), 1023);

  console::couti << "NVML device name: `" << std::string(device_name.data()) << "`";

  nvmlMemory_t memory_info;
  NVML_CALL(nvmlDeviceGetMemoryInfo, device, &memory_info);

  console::couti << "NVML Memory Used: " << memory_info.used << " / " << memory_info.total;
  console::couti << "NVML Memory Used: " << memory_info.used / 1024 / 1024 << " / " << memory_info.total / 1024 / 1024;
  console::couti << "NVML Memory Free: " << memory_info.free << " / " << memory_info.total;
  console::couti << "NVML Memory Free: " << memory_info.free / 1024 / 1024 << " / " << memory_info.total / 1024 / 1024;

  std::vector<char> driver_version;
  driver_version.resize(1024, '\0');
  NVML_CALL(nvmlSystemGetDriverVersion, driver_version.data(),1023);

  console::couti << "NVML GPU Driver Version: `" << std::string(driver_version.data()) << "`";

  int cuda_driver_version = 0;

  NVML_CALL(nvmlSystemGetCudaDriverVersion, &cuda_driver_version);
  console::couti << "NVML CUDA Driver Version: " << NVML_CUDA_DRIVER_VERSION_MAJOR(cuda_driver_version) << "." << NVML_CUDA_DRIVER_VERSION_MINOR(cuda_driver_version);

  NVML_CALL(nvmlSystemGetCudaDriverVersion_v2, &cuda_driver_version);
  console::couti << "NVML CUDA Driver Version 2: " << NVML_CUDA_DRIVER_VERSION_MAJOR(cuda_driver_version) << "." << NVML_CUDA_DRIVER_VERSION_MINOR(cuda_driver_version);


  // IA_NVML_CALL(nvmlShutdown())

  // console::couti << "NVML Shut down";

  // int         deviceCount = 0;
  // cudaError_t error = cudaGetDeviceCount(&deviceCount);

  // if (error != cudaSuccess) {
  //   std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  //   return -1;
  // }

  // std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

  // for (int i = 0; i < deviceCount; ++i) {
  //   cudaDeviceProp deviceProp;
  //   cudaGetDeviceProperties(&deviceProp, i);
  //   std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
  // }

  return 0;
}

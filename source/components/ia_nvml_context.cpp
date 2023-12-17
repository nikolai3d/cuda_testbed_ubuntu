#include "ia_nvml_context.h"

#include <dlfcn.h>

#define NVML_CALL(NVML_FUNCTION, ...) IA_NVML_CALL(_nvml_function_table, NVML_FUNCTION, __VA_ARGS__)
namespace ia_nvml {

NVMLContext::NVMLContext() : _thread_id(std::this_thread::get_id()) {
  // Load NVML library
  _nvml_library_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!_nvml_library_handle) {
    return;
  }

  // Load NVML functions
  _nvml_function_table.initialize_nvml_function_pointers(_nvml_library_handle);
  
  NVML_CALL(nvmlInit_v2);
}

NVMLContext::~NVMLContext() {
  if (!_nvml_library_handle) {
    return;
  }

  NVML_CALL(nvmlShutdown);
  dlclose(_nvml_library_handle);
}

}  // namespace ia_nvml
#include "ia_nvml_context.h"

#include <dlfcn.h>

namespace ia_nvml {

NVMLContext::NVMLContext() : _thread_id(std::this_thread::get_id()) {
  // Load NVML library
  _nvml_library_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!_nvml_library_handle) {
    return;
  }

  // Load NVML functions
  _nvml_function_table.initialize_nvml_function_pointers(_nvml_library_handle);
}

NVMLContext::~NVMLContext() {
  dlclose(_nvml_library_handle);
}

}  // namespace ia_nvml
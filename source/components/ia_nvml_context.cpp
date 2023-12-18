#include "ia_nvml_context.h"

#include <dlfcn.h>

namespace ia_nvml {

NVMLContext::NVMLContext() : _thread_id(std::this_thread::get_id()), _is_initialized(false) {
  // Load NVML Library and query the functions
  const bool init_dll = _nvml_function_table.initialize_nvml_function_pointers();

  if (!init_dll) {
    return;
  }

  // Attempt to initialize NVML
  const auto init_nvml = IA_NVML_CALL_VERBOSE(_nvml_function_table, nvmlInit_v2);

  if (init_nvml != NVML_SUCCESS) {
    return;
  }

  _is_initialized = true;
}

NVMLContext::~NVMLContext() {
  IA_NVML_CALL(_nvml_function_table, nvmlShutdown);
}

}  // namespace ia_nvml
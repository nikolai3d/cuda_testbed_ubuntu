#include "ia_nvml_context.h"

#include <dlfcn.h>

namespace ia_nvml {

NVMLContext::NVMLContext() : _thread_id(std::this_thread::get_id()) {
  // Load NVML Library and query the functions
  _nvml_function_table.initialize_nvml_function_pointers();

  IA_NVML_CALL(_nvml_function_table, nvmlInit_v2);
}

NVMLContext::~NVMLContext() {
  IA_NVML_CALL(_nvml_function_table, nvmlShutdown);
}

}  // namespace ia_nvml
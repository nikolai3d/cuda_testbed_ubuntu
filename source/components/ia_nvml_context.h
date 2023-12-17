#pragma once
#include "ia_nvml.hpp"

#include <memory>

namespace ia_nvml {
class NVMLContext {
  NVMLFunctionTable         _nvml_function_table;
  void*                     _nvml_library_handle = nullptr;
  std::thread_id            _thread_id;  // To avoid multiple threads issue, make sure NVML context can only be called from one thread

 public:
  NVMLContext();

  ~NVMLContext();

  NVMLContext(const NVMLContext&) = delete;
  NVMLContext& operator=(const NVMLContext&) = delete;
  NVMLContext(NVMLContext&&) = default;
  NVMLContext& operator=(NVMLContext&&) = default;
};

std::unique_ptr<ia_nvml::NVMLContext> create_nvml_context() {
  return std::make_unique<ia_nvml::NVMLContext>();
}

}  // namespace ia_nvml

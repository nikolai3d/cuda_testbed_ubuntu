#pragma once
#include "ia_nvml.hpp"

#include <memory>
#include <thread>

namespace ia_nvml {
class NVMLContext {
public:
  NVMLFunctionTable         _nvml_function_table;
  std::thread::id           _thread_id;  // To avoid multiple threads issue, make sure NVML context can only be called from one thread
  bool                      _is_initialized = false; // True if DLL is loaded and NVML has been successfully initialized
 public:
  [[nodiscard]] bool is_ok() const {
    return _is_initialized;
  } 
  NVMLContext();

  ~NVMLContext();

  NVMLContext(const NVMLContext&) = delete;
  NVMLContext& operator=(const NVMLContext&) = delete;
  NVMLContext(NVMLContext&&) = default;
  NVMLContext& operator=(NVMLContext&&) = default;
};


}  // namespace ia_nvml

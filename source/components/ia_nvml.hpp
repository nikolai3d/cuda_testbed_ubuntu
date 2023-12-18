#pragma once

#include <nvml.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>


namespace ia_nvml {

/**
 * @brief Class representing the NVML function table.
 *
 * This class encapsulates the function pointers for various NVML functions, providing a convenient way to access and call NVML functions dynamically.
 * It also holds the DLL handle for the NVML library.
 * Each function pointer is declared using the IA_DECLARE_FPTR macro.
 *
 * - On instantiation, everything is initialized to nullptr.
 * - The initialize_nvml_function_pointers function opens the DLL and initializes the function pointers using the provided NVML library handle.
 * - If the DLL handle has been initialized, the destructor would close/free it and unload the DLL.
 * - No thread safety is provided, so make sure to call the functions from a single thread or implement your own thread safety mechanisms.
 *
 * Calling NVML functions:
 *
 * - The NVML function pointers are public, so they can be accessed directly, so you can call them like this:
 *
 *  NVMLFunctionTable ft;
 *  ft.initialize_nvml_function_pointers();
 *  ft.pfn_nvmlInit_v2();
 *  ft.pfn_nvmlDeviceGetCount_v2(&device_count);
 *
 *
 * - You can also use the provided call macro to call NVML functions, some versions of the macro provide some additional error checking and reporting. There are four versions of the macro:
 * - IA_NVML_CALL_NO_CHECK(...) - the no-check version. It returns the nvmlReturn_t value returned by the NVML function. Does not check if the function pointer is null, so it may crash.
 * - IA_NVML_CALL(...) - the silent version. It returns the nvmlReturn_t value returned by the NVML function or NVML_ERROR_UNINITIALIZED if the function pointer is null.
 * - IA_NVML_CALL_VERBOSE(...) - the verbose version, which prints an error message to stderr if the function call fails or null. It returns the nvmlReturn_t value returned by the NVML function or NVML_ERROR_UNINITIALIZED if the function pointer is null.
 * - IA_NVML_CALL_THROW(...) - the throwing version, which throws an std::runtime_error if the function call fails or null. It is guaranteed to returns the nvmlReturn_t value of NVML_SUCCESS if the function call succeeds.
 *
 *  NVMLFunctionTable ft;
 *  ft.initialize_nvml_function_pointers();
 *  IA_NVML_CALL(ft, nvmlInit_v2);
 *  IA_NVML_CALL(ft, nvmlDeviceGetCount_v2, &device_count);
 * 
 * - The types of the arguments passed to the NVML functions are checked against functions' signatures at compile time, make sure to match the types exactly.
 * - If you pass the wrong arguments to the NVML functions, you will get a compile-time error.
 * E.g.: 
 *    int device_count; IA_NVML_CALL(ft, nvmlDeviceGetCount_v2, &device_count); // This will generate a compile-time error, because the second argument should be a unsigned int pointer.
 *    unsigned int device_count; IA_NVML_CALL(ft, nvmlDeviceGetCount_v2, &device_count); // This is correct.
 */

class NVMLFunctionTable {
  // DLL/SO handle
  void *_nvml_dll_handle = nullptr;

 public:

// Macro for declaring function pointer
#ifndef IA_DECLARE_FPTR
#define IA_DECLARE_FPTR(name, rettype, arglist) rettype(*pfn_##name) arglist = nullptr;
#endif
  // NVML Function Pointers:
  // Reference: https://docs.nvidia.com/deploy/nvml-api/index.html
  IA_DECLARE_FPTR(nvmlInitWithFlags, nvmlReturn_t, (unsigned int));
  IA_DECLARE_FPTR(nvmlInit_v2, nvmlReturn_t, (void));
  IA_DECLARE_FPTR(nvmlShutdown, nvmlReturn_t, (void));

  IA_DECLARE_FPTR(nvmlSystemGetCudaDriverVersion, nvmlReturn_t, (int *));
  IA_DECLARE_FPTR(nvmlSystemGetCudaDriverVersion_v2, nvmlReturn_t, (int *));
  IA_DECLARE_FPTR(nvmlSystemGetDriverVersion, nvmlReturn_t, (char *, unsigned int));
  IA_DECLARE_FPTR(nvmlSystemGetNVMLVersion, nvmlReturn_t, (char *, unsigned int));
  IA_DECLARE_FPTR(nvmlSystemGetProcessName, nvmlReturn_t, (unsigned int, char *, unsigned int));

  IA_DECLARE_FPTR(nvmlDeviceGetAPIRestriction, nvmlReturn_t, (nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetAdaptiveClockInfoStatus, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetApplicationsClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetArchitecture, nvmlReturn_t, (nvmlDevice_t, nvmlDeviceArchitecture_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetAttributes_v2, nvmlReturn_t, (nvmlDevice_t, nvmlDeviceAttributes_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetBAR1MemoryInfo, nvmlReturn_t, (nvmlDevice_t, nvmlBAR1Memory_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetBoardId, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetBoardPartNumber, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetBrand, nvmlReturn_t, (nvmlDevice_t, nvmlBrandType_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetBridgeChipInfo, nvmlReturn_t, (nvmlDevice_t, nvmlBridgeChipHierarchy_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetClockInfo, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetComputeMode, nvmlReturn_t, (nvmlDevice_t, nvmlComputeMode_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetComputeRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetCudaComputeCapability, nvmlReturn_t, (nvmlDevice_t, int *, int *));
  IA_DECLARE_FPTR(nvmlDeviceGetCurrPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetCurrPcieLinkWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetCurrentClocksThrottleReasons, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetDecoderUtilization, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetDefaultApplicationsClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetDefaultEccMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetDetailedEccErrors, nvmlReturn_t, (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetDisplayActive, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetDisplayMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetDriverModel, nvmlReturn_t, (nvmlDevice_t, nvmlDriverModel_t *, nvmlDriverModel_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetEccMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetEncoderCapacity, nvmlReturn_t, (nvmlDevice_t, nvmlEncoderType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetEncoderSessions, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlEncoderSessionInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetEncoderStats, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetEncoderUtilization, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetEnforcedPowerLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetFBCSessions, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlFBCSessionInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetFBCStats, nvmlReturn_t, (nvmlDevice_t, nvmlFBCStats_t *));
  // IA_DECLARE_FPTR(nvmlDeviceGetFanControlPolicy_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetFanSpeed_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetGpuMaxPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetGpuOperationMode, nvmlReturn_t, (nvmlDevice_t, nvmlGpuOperationMode_t *, nvmlGpuOperationMode_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetGraphicsRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetHandleByIndex_v2, nvmlReturn_t, (unsigned int, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetHandleByPciBusId_v2, nvmlReturn_t, (const char *, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetHandleBySerial, nvmlReturn_t, (const char *, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetHandleByUUID, nvmlReturn_t, (const char *, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetIndex, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetInforomConfigurationChecksum, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetInforomImageVersion, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetInforomVersion, nvmlReturn_t, (nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetIrqNum, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMPSComputeRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetMaxClockInfo, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMaxCustomerBoostClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMaxPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMaxPcieLinkWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMemoryBusWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMemoryErrorCounter, nvmlReturn_t, (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetMemoryInfo, nvmlReturn_t, (nvmlDevice_t, nvmlMemory_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetMinMaxFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMinorNumber, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetMultiGpuBoard, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetName, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetNumFans, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetNumGpuCores, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetP2PStatus, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPciInfo_v3, nvmlReturn_t, (nvmlDevice_t, nvmlPciInfo_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPcieLinkMaxSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPcieReplayCounter, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPcieSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPcieThroughput, nvmlReturn_t, (nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPerformanceState, nvmlReturn_t, (nvmlDevice_t, nvmlPstates_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPersistenceMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerManagementDefaultLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerManagementLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerManagementLimitConstraints, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerManagementMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerSource, nvmlReturn_t, (nvmlDevice_t, nvmlPowerSource_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerState, nvmlReturn_t, (nvmlDevice_t, nvmlPstates_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetPowerUsage, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetRemappedRows, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetRetiredPages, nvmlReturn_t, (nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetRetiredPagesPendingStatus, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetRetiredPages_v2, nvmlReturn_t, (nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetRowRemapperHistogram, nvmlReturn_t, (nvmlDevice_t, nvmlRowRemapperHistogramValues_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetSamples, nvmlReturn_t, (nvmlDevice_t, nvmlSamplingType_t, unsigned long long, nvmlValueType_t *, unsigned int *, nvmlSample_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetSerial, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetSupportedClocksThrottleReasons, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetSupportedGraphicsClocks, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetSupportedMemoryClocks, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetTargetFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetTemperature, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetTemperatureThreshold, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int *));
  IA_DECLARE_FPTR(nvmlDeviceGetThermalSettings, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlGpuThermalSettings_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetTopologyCommonAncestor, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, nvmlGpuTopologyLevel_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetTopologyNearestGpus, nvmlReturn_t, (nvmlDevice_t, nvmlGpuTopologyLevel_t, unsigned int *, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetTotalEccErrors, nvmlReturn_t, (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetTotalEnergyConsumption, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
  IA_DECLARE_FPTR(nvmlDeviceGetUUID, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetUtilizationRates, nvmlReturn_t, (nvmlDevice_t, nvmlUtilization_t *));
  IA_DECLARE_FPTR(nvmlDeviceGetVbiosVersion, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceGetViolationStatus, nvmlReturn_t, (nvmlDevice_t, nvmlPerfPolicyType_t, nvmlViolationTime_t *));
  IA_DECLARE_FPTR(nvmlDeviceOnSameBoard, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, int *));
  IA_DECLARE_FPTR(nvmlDeviceResetApplicationsClocks, nvmlReturn_t, (nvmlDevice_t));
  IA_DECLARE_FPTR(nvmlDeviceSetAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t));
  IA_DECLARE_FPTR(nvmlDeviceSetDefaultAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t, unsigned int));
  IA_DECLARE_FPTR(nvmlDeviceSetDefaultFanSpeed_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int));
  // IA_DECLARE_FPTR(nvmlDeviceSetFanControlPolicy, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t));
  IA_DECLARE_FPTR(nvmlDeviceSetTemperatureThreshold, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureThresholds_t, int *));
  IA_DECLARE_FPTR(nvmlDeviceValidateInforom, nvmlReturn_t, (nvmlDevice_t));
  IA_DECLARE_FPTR(nvmlSystemGetTopologyGpuSet, nvmlReturn_t, (unsigned int, unsigned int *, nvmlDevice_t *));
  IA_DECLARE_FPTR(nvmlVgpuInstanceGetMdevUUID, nvmlReturn_t, (nvmlVgpuInstance_t, char *, unsigned int));

  IA_DECLARE_FPTR(nvmlErrorString, const char *, (nvmlReturn_t));

#undef IA_DECLARE_FPTR
  /**
   * @brief Initializes the function pointers for NVML (NVIDIA Management Library).
   *
   * This function initializes the necessary function pointers for interacting with NVML.
   * It should be called before any other NVML-related operations.
   *
   * @return true if the DLL load is successful, false otherwise. Does not check if the actual function pointers are succesfully initialized.
   */
  bool initialize_nvml_function_pointers();

  NVMLFunctionTable() = default;
  // Non-copyable, non-movable
  NVMLFunctionTable(const NVMLFunctionTable &) = delete;
  NVMLFunctionTable &operator=(const NVMLFunctionTable &) = delete;
  NVMLFunctionTable(NVMLFunctionTable &&) = delete;
  NVMLFunctionTable &operator=(NVMLFunctionTable &&) = delete;
  /**
   * @brief The destructor frees up the NVML library handle, if it has been initialized.
   */
  ~NVMLFunctionTable();
};

namespace detail {
/**
 * Helper template that invokes a function with the given arguments. Arguments are passed in as a tuple, get unpacked and forwarded to the function.
 * Don't use this function directly, it's only supposed to be used by the call_function_with_arguments_tuple function.
 * It performs argument forwarding and checks if the function can be invoked with the given arguments.
 * If the function cannot be invoked with the given arguments, a compile-time error is generated.
 *
 * @tparam Function The type of the callable object / function.
 * @tparam ArgTuple The type of the tuple containing the arguments.
 * @tparam I The indices of the arguments in the tuple.
 * @param func The callable object / function to be invoked.
 * @param t The tuple containing the callable object/function arguments.
 * @return The result of invoking the function with the given arguments.
 * @throws std::bad_function_call if the function cannot be invoked with the given arguments.
 */
template <typename Function, typename ArgTuple, std::size_t... I>
decltype(auto) invoke_helper(Function &&func, ArgTuple &&tuple, std::index_sequence<I...>) {
  static_assert(std::is_invocable_v<Function, decltype(std::get<I>(std::forward<ArgTuple>(tuple)))...>, "Function cannot be invoked with the given arguments, check the NVML function signature and the input argument types");
  return std::invoke(std::forward<Function>(func), std::get<I>(std::forward<ArgTuple>(tuple))...);
}

using error_string_provider_t = std::function<const char *(nvmlReturn_t)>;
enum class nvml_call_type_t : std::int32_t { NoCheck = 0, Silent, Verbose, Throw };

template <nvml_call_type_t CallType>
void handle_nvml_error(const std::string &i_error_message) {
  static_assert(CallType == nvml_call_type_t::Throw || CallType == nvml_call_type_t::Verbose, "This function should only be called for Throw or Verbose NVML call types");
  if constexpr (CallType == nvml_call_type_t::Throw) {
    throw std::runtime_error(i_error_message);
  } else {
    std::cerr << i_error_message << std::endl;
  }
}

/**
 * Calls an NVML function with arguments provided as a tuple. (Don't use this function directly, use the IA_NVML_CALL macro instead.). If the function call fails, an error message is printed to stderr.
 * Cannot be used to call NVML functions that return something other than nvmlReturn_t: this is enforced by a static_assert and would generate a compile-time error.
 * @tparam CallType The type of the NVML call. Can be nvml_call_type_t::NoCheck, nvml_call_type_t::Silent, nvml_call_type_t::Verbose or nvml_call_type_t::Throw.
 * @param file The name of the file where the function is called. (for error reporting)
 * @param line The line number where the function is called. (for error reporting)
 * @param function_call The name of the function being called. (for error reporting)
 * @param func NVML The function to be called.
 * @param t Arguments to be passed to the NVML function, in the form of a tuple
 * @return nvmlReturn_t The return value of the NVML function call.
 */
template <nvml_call_type_t CallType, typename Function, typename ArgTuple>
nvmlReturn_t call_function_with_arguments_tuple(const char *file, const std::int64_t line, const char *function_call, error_string_provider_t error_string_provider, Function &&func, ArgTuple &&tuple) {
  if constexpr (CallType == nvml_call_type_t::NoCheck) {
    constexpr auto tuple_size = std::tuple_size_v<std::remove_reference_t<ArgTuple>>;
    return invoke_helper(std::forward<Function>(func), std::forward<ArgTuple>(tuple), std::make_index_sequence<tuple_size>{});
  }

  if (!func) {
    if constexpr (CallType == nvml_call_type_t::Throw || CallType == nvml_call_type_t::Verbose) {
      std::stringstream ss;
      ss << "ERROR: NVML function " << function_call << " at line " << line << " of file " << file << " is NULL: either its pointer had not been initialized, or is missing from the DLL/SO.\n";

      handle_nvml_error<CallType>(ss.str());
    }

    return NVML_ERROR_UNINITIALIZED;
  }

  constexpr auto tuple_size = std::tuple_size_v<std::remove_reference_t<ArgTuple>>;
  const auto     status = invoke_helper(std::forward<Function>(func), std::forward<ArgTuple>(tuple), std::make_index_sequence<tuple_size>{});
  // Static assert to check return type
  static_assert(std::is_same_v<std::remove_const_t<decltype(status)>, nvmlReturn_t>, "with this wrapper, you can only NVML functions that return the nvmlReturn_t type");

  if (status != NVML_SUCCESS) {
    if constexpr (CallType == nvml_call_type_t::Throw || CallType == nvml_call_type_t::Verbose) {
      std::stringstream ss;
      ss << "ERROR: CUDA NVML call " << function_call << " at line " << line << " of file " << file << " failed with error \"" << error_string_provider(status) << "\" (error code " << status << ").\n";
      handle_nvml_error<CallType>(ss.str());
    }
  }
  return static_cast<nvmlReturn_t>(status);
}
}  // namespace detail

/**
 * @brief Macro to call a NVML function from a function table
 * To call a NVML function, use the IA_NVML_CALL macro, passing the function table and the function name, followed by the arguments to be passed to the function.
 * E.g.:
 * NVMLFunctionTable ft;
 * ft.initialize_nvml_function_pointers();
 * ...
 * nvmlDevice_t device;
 * std::uint32_t device_count = 0;
 * IA_NVML_CALL(ft, nvmlInit_v2);
 * IA_NVML_CALL(ft, nvmlDeviceGetCount_v2, &device_count);
 * IA_NVML_CALL(ft, nvmlDeviceGetHandleByIndex_v2, 0, &device);
 * 
 * There are four versions of the macro:
 * - IA_NVML_CALL_NO_CHECK(...) - the no-check version, fastest. It returns the nvmlReturn_t value returned by the NVML function. Does not check if the function pointer is null, so it may crash.
 * - IA_NVML_CALL(...) - the silent version. It returns the nvmlReturn_t value returned by the NVML function or NVML_ERROR_UNINITIALIZED if the function pointer is null.
 * - IA_NVML_CALL_VERBOSE(...) - the verbose version, which prints an error message to stderr if the function call fails or null. It returns the nvmlReturn_t value returned by the NVML function or NVML_ERROR_UNINITIALIZED if the function pointer is null.
 * - IA_NVML_CALL_THROW(...) - the throwing version, which throws an std::runtime_error if the function call fails or null. It is guaranteed to returns the nvmlReturn_t value of NVML_SUCCESS if the function call succeeds.
 */
#define _IA_NVML_CALL(NVML_CALL_TYPE, NVML_FUNCTION_TABLE, NVML_FUNCTION, ...) \
  ia_nvml::detail::call_function_with_arguments_tuple<NVML_CALL_TYPE>(__FILE__, __LINE__, #NVML_FUNCTION, NVML_FUNCTION_TABLE.pfn_nvmlErrorString, NVML_FUNCTION_TABLE.pfn_##NVML_FUNCTION, std::make_tuple(__VA_ARGS__))

#define IA_NVML_CALL_NO_CHECK(NVML_FUNCTION_TABLE, NVML_FUNCTION, ...) _IA_NVML_CALL(ia_nvml::detail::nvml_call_type_t::NoCheck, NVML_FUNCTION_TABLE, NVML_FUNCTION, __VA_ARGS__)
#define IA_NVML_CALL(NVML_FUNCTION_TABLE, NVML_FUNCTION, ...) _IA_NVML_CALL(ia_nvml::detail::nvml_call_type_t::Silent, NVML_FUNCTION_TABLE, NVML_FUNCTION, __VA_ARGS__)
#define IA_NVML_CALL_VERBOSE(NVML_FUNCTION_TABLE, NVML_FUNCTION, ...) _IA_NVML_CALL(ia_nvml::detail::nvml_call_type_t::Verbose, NVML_FUNCTION_TABLE, NVML_FUNCTION, __VA_ARGS__)
#define IA_NVML_CALL_THROW(NVML_FUNCTION_TABLE, NVML_FUNCTION, ...) _IA_NVML_CALL(ia_nvml::detail::nvml_call_type_t::Throw, NVML_FUNCTION_TABLE, NVML_FUNCTION, __VA_ARGS__)


}  // namespace ia_nvml

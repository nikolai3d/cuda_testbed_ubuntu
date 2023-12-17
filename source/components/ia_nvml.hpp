#pragma once

#include <nvml.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <tuple>
#include <utility>
#include <type_traits>

// Macro for declaring function pointer
#ifndef IA_DECLARE_FPTR
#define IA_DECLARE_FPTR(name, rettype, arglist) rettype (*pfn_##name) arglist = nullptr;
#endif

namespace ia_nvml {

struct NVMLFunctionTable {
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

  void initialize_nvml_function_pointers(void *i_dll_handle);

  static NVMLFunctionTable &instance();
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

/**
 * Calls an NVML function with arguments provided as a tuple. (Don't use this function directly, use the IA_NVML_CALL macro instead.). If the function call fails, an error message is printed to stderr.
 *
 * @param file The name of the file where the function is called. (for error reporting)
 * @param line The line number where the function is called. (for error reporting)
 * @param function_call The name of the function being called. (for error reporting)
 * @param func NVML The function to be called.
 * @param t Arguments to be passed to the NVML function, in the form of a tuple 
 * @return void
 */
template <typename Function, typename ArgTuple>
decltype(auto) call_function_with_arguments_tuple(const char *file, const std::int64_t line, const char *function_call, Function &&func, ArgTuple &&tuple) {
  constexpr auto tuple_size = std::tuple_size_v<std::remove_reference_t<ArgTuple>>;
  const auto     status = invoke_helper(std::forward<Function>(func), std::forward<ArgTuple>(tuple), std::make_index_sequence<tuple_size>{});
 // Static assert to check return type
  static_assert(std::is_same_v<std::remove_const_t<decltype(status)>, nvmlReturn_t>, "with this wrapper, you can only NVML functions that return the nvmlReturn_t type");

  if (status != NVML_SUCCESS) {
    std::stringstream ss;
    ss << "ERROR: CUDA NVML call " << function_call << " at line " << line << " of file " << file << " failed with error \"" << NVMLFunctionTable::instance().pfn_nvmlErrorString(status) << "\" (error code " << status << ").\n";
    std::cerr << ss.str() << std::endl;
  }
}
}  // namespace detail

#define IA_NVML_CALL(NVML_FUNCTION, ...) ia_nvml::detail::call_function_with_arguments_tuple(__FILE__, __LINE__, #NVML_FUNCTION, ia_nvml::NVMLFunctionTable::instance().pfn_##NVML_FUNCTION, std::make_tuple(__VA_ARGS__))

}  // namespace ia_nvml

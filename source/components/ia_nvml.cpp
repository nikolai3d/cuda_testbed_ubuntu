#include "ia_nvml.hpp"
#include <dlfcn.h>


#include <initializer_list>
#include <iostream>


namespace {


struct Symbol {
  void      **_ppfn;
  const char *_function_name;
};

}  // namespace

namespace ia_nvml {

NVMLFunctionTable &NVMLFunctionTable::instance() {
  static NVMLFunctionTable s_instance;
  return s_instance;
}

void NVMLFunctionTable::initialize_nvml_function_pointers() {
  void *libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (libhandle == nullptr) {
    std::cout << "Failed to open libnvidia-ml.so.1" << std::endl;
    return;
  }

  std::initializer_list<Symbol> symbols = {

      {(void **)&pfn_nvmlInitWithFlags, "nvmlInitWithFlags"},
      {(void **)&pfn_nvmlInit_v2, "nvmlInit_v2"},
      {(void **)&pfn_nvmlShutdown, "nvmlShutdown"},

      {(void **)&pfn_nvmlSystemGetCudaDriverVersion, "nvmlSystemGetCudaDriverVersion"},
      {(void **)&pfn_nvmlSystemGetCudaDriverVersion_v2, "nvmlSystemGetCudaDriverVersion_v2"},
      {(void **)&pfn_nvmlSystemGetDriverVersion, "nvmlSystemGetDriverVersion"},
      {(void **)&pfn_nvmlSystemGetNVMLVersion, "nvmlSystemGetNVMLVersion"},
      {(void **)&pfn_nvmlSystemGetProcessName, "nvmlSystemGetProcessName"},

      {(void **)&pfn_nvmlDeviceGetAPIRestriction, "nvmlDeviceGetAPIRestriction"},
      {(void **)&pfn_nvmlDeviceGetAdaptiveClockInfoStatus, "nvmlDeviceGetAdaptiveClockInfoStatus"},
      {(void **)&pfn_nvmlDeviceGetApplicationsClock, "nvmlDeviceGetApplicationsClock"},
      {(void **)&pfn_nvmlDeviceGetArchitecture, "nvmlDeviceGetArchitecture"},
      {(void **)&pfn_nvmlDeviceGetAttributes_v2, "nvmlDeviceGetAttributes_v2"},
      {(void **)&pfn_nvmlDeviceGetAutoBoostedClocksEnabled, "nvmlDeviceGetAutoBoostedClocksEnabled"},
      {(void **)&pfn_nvmlDeviceGetBAR1MemoryInfo, "nvmlDeviceGetBAR1MemoryInfo"},
      {(void **)&pfn_nvmlDeviceGetBoardId, "nvmlDeviceGetBoardId"},
      {(void **)&pfn_nvmlDeviceGetBoardPartNumber, "nvmlDeviceGetBoardPartNumber"},
      {(void **)&pfn_nvmlDeviceGetBrand, "nvmlDeviceGetBrand"},
      {(void **)&pfn_nvmlDeviceGetBridgeChipInfo, "nvmlDeviceGetBridgeChipInfo"},
      {(void **)&pfn_nvmlDeviceGetClock, "nvmlDeviceGetClock"},
      {(void **)&pfn_nvmlDeviceGetClockInfo, "nvmlDeviceGetClockInfo"},
      {(void **)&pfn_nvmlDeviceGetComputeMode, "nvmlDeviceGetComputeMode"},
      {(void **)&pfn_nvmlDeviceGetComputeRunningProcesses_v3, "nvmlDeviceGetComputeRunningProcesses_v3"},
      {(void **)&pfn_nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2"},
      {(void **)&pfn_nvmlDeviceGetCudaComputeCapability, "nvmlDeviceGetCudaComputeCapability"},
      {(void **)&pfn_nvmlDeviceGetCurrPcieLinkGeneration, "nvmlDeviceGetCurrPcieLinkGeneration"},
      {(void **)&pfn_nvmlDeviceGetCurrPcieLinkWidth, "nvmlDeviceGetCurrPcieLinkWidth"},
      {(void **)&pfn_nvmlDeviceGetCurrentClocksThrottleReasons, "nvmlDeviceGetCurrentClocksThrottleReasons"},
      {(void **)&pfn_nvmlDeviceGetDecoderUtilization, "nvmlDeviceGetDecoderUtilization"},
      {(void **)&pfn_nvmlDeviceGetDefaultApplicationsClock, "nvmlDeviceGetDefaultApplicationsClock"},
      {(void **)&pfn_nvmlDeviceGetDefaultEccMode, "nvmlDeviceGetDefaultEccMode"},
      {(void **)&pfn_nvmlDeviceGetDetailedEccErrors, "nvmlDeviceGetDetailedEccErrors"},
      {(void **)&pfn_nvmlDeviceGetDisplayActive, "nvmlDeviceGetDisplayActive"},
      {(void **)&pfn_nvmlDeviceGetDisplayMode, "nvmlDeviceGetDisplayMode"},
      {(void **)&pfn_nvmlDeviceGetDriverModel, "nvmlDeviceGetDriverModel"},
      {(void **)&pfn_nvmlDeviceGetEccMode, "nvmlDeviceGetEccMode"},
      {(void **)&pfn_nvmlDeviceGetEncoderCapacity, "nvmlDeviceGetEncoderCapacity"},
      {(void **)&pfn_nvmlDeviceGetEncoderSessions, "nvmlDeviceGetEncoderSessions"},
      {(void **)&pfn_nvmlDeviceGetEncoderStats, "nvmlDeviceGetEncoderStats"},
      {(void **)&pfn_nvmlDeviceGetEncoderUtilization, "nvmlDeviceGetEncoderUtilization"},
      {(void **)&pfn_nvmlDeviceGetEnforcedPowerLimit, "nvmlDeviceGetEnforcedPowerLimit"},
      {(void **)&pfn_nvmlDeviceGetFBCSessions, "nvmlDeviceGetFBCSessions"},
      {(void **)&pfn_nvmlDeviceGetFBCStats, "nvmlDeviceGetFBCStats"},
      //     {(void **)&pfn_nvmlDeviceGetFanControlPolicy_v2, "nvmlDeviceGetFanControlPolicy_v2"},
      {(void **)&pfn_nvmlDeviceGetFanSpeed, "nvmlDeviceGetFanSpeed"},
      {(void **)&pfn_nvmlDeviceGetFanSpeed_v2, "nvmlDeviceGetFanSpeed_v2"},
      {(void **)&pfn_nvmlDeviceGetGpuMaxPcieLinkGeneration, "nvmlDeviceGetGpuMaxPcieLinkGeneration"},
      {(void **)&pfn_nvmlDeviceGetGpuOperationMode, "nvmlDeviceGetGpuOperationMode"},
      {(void **)&pfn_nvmlDeviceGetGraphicsRunningProcesses_v3, "nvmlDeviceGetGraphicsRunningProcesses_v3"},
      {(void **)&pfn_nvmlDeviceGetHandleByIndex_v2, "nvmlDeviceGetHandleByIndex_v2"},
      {(void **)&pfn_nvmlDeviceGetHandleByPciBusId_v2, "nvmlDeviceGetHandleByPciBusId_v2"},
      {(void **)&pfn_nvmlDeviceGetHandleBySerial, "nvmlDeviceGetHandleBySerial"},
      {(void **)&pfn_nvmlDeviceGetHandleByUUID, "nvmlDeviceGetHandleByUUID"},
      {(void **)&pfn_nvmlDeviceGetIndex, "nvmlDeviceGetIndex"},
      {(void **)&pfn_nvmlDeviceGetInforomConfigurationChecksum, "nvmlDeviceGetInforomConfigurationChecksum"},
      {(void **)&pfn_nvmlDeviceGetInforomImageVersion, "nvmlDeviceGetInforomImageVersion"},
      {(void **)&pfn_nvmlDeviceGetInforomVersion, "nvmlDeviceGetInforomVersion"},
      {(void **)&pfn_nvmlDeviceGetIrqNum, "nvmlDeviceGetIrqNum"},
      {(void **)&pfn_nvmlDeviceGetMPSComputeRunningProcesses_v3, "nvmlDeviceGetMPSComputeRunningProcesses_v3"},
      {(void **)&pfn_nvmlDeviceGetMaxClockInfo, "nvmlDeviceGetMaxClockInfo"},
      {(void **)&pfn_nvmlDeviceGetMaxCustomerBoostClock, "nvmlDeviceGetMaxCustomerBoostClock"},
      {(void **)&pfn_nvmlDeviceGetMaxPcieLinkGeneration, "nvmlDeviceGetMaxPcieLinkGeneration"},
      {(void **)&pfn_nvmlDeviceGetMaxPcieLinkWidth, "nvmlDeviceGetMaxPcieLinkWidth"},
      {(void **)&pfn_nvmlDeviceGetMemoryBusWidth, "nvmlDeviceGetMemoryBusWidth"},
      {(void **)&pfn_nvmlDeviceGetMemoryErrorCounter, "nvmlDeviceGetMemoryErrorCounter"},
      {(void **)&pfn_nvmlDeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo"},
      {(void **)&pfn_nvmlDeviceGetMinMaxFanSpeed, "nvmlDeviceGetMinMaxFanSpeed"},
      {(void **)&pfn_nvmlDeviceGetMinorNumber, "nvmlDeviceGetMinorNumber"},
      {(void **)&pfn_nvmlDeviceGetMultiGpuBoard, "nvmlDeviceGetMultiGpuBoard"},
      {(void **)&pfn_nvmlDeviceGetName, "nvmlDeviceGetName"},
      {(void **)&pfn_nvmlDeviceGetNumFans, "nvmlDeviceGetNumFans"},
      {(void **)&pfn_nvmlDeviceGetNumGpuCores, "nvmlDeviceGetNumGpuCores"},
      {(void **)&pfn_nvmlDeviceGetP2PStatus, "nvmlDeviceGetP2PStatus"},
      {(void **)&pfn_nvmlDeviceGetPciInfo_v3, "nvmlDeviceGetPciInfo_v3"},
      {(void **)&pfn_nvmlDeviceGetPcieLinkMaxSpeed, "nvmlDeviceGetPcieLinkMaxSpeed"},
      {(void **)&pfn_nvmlDeviceGetPcieReplayCounter, "nvmlDeviceGetPcieReplayCounter"},
      {(void **)&pfn_nvmlDeviceGetPcieSpeed, "nvmlDeviceGetPcieSpeed"},
      {(void **)&pfn_nvmlDeviceGetPcieThroughput, "nvmlDeviceGetPcieThroughput"},
      {(void **)&pfn_nvmlDeviceGetPerformanceState, "nvmlDeviceGetPerformanceState"},
      {(void **)&pfn_nvmlDeviceGetPersistenceMode, "nvmlDeviceGetPersistenceMode"},
      {(void **)&pfn_nvmlDeviceGetPowerManagementDefaultLimit, "nvmlDeviceGetPowerManagementDefaultLimit"},
      {(void **)&pfn_nvmlDeviceGetPowerManagementLimit, "nvmlDeviceGetPowerManagementLimit"},
      {(void **)&pfn_nvmlDeviceGetPowerManagementLimitConstraints, "nvmlDeviceGetPowerManagementLimitConstraints"},
      {(void **)&pfn_nvmlDeviceGetPowerManagementMode, "nvmlDeviceGetPowerManagementMode"},
      {(void **)&pfn_nvmlDeviceGetPowerSource, "nvmlDeviceGetPowerSource"},
      {(void **)&pfn_nvmlDeviceGetPowerState, "nvmlDeviceGetPowerState"},
      {(void **)&pfn_nvmlDeviceGetPowerUsage, "nvmlDeviceGetPowerUsage"},
      {(void **)&pfn_nvmlDeviceGetRemappedRows, "nvmlDeviceGetRemappedRows"},
      {(void **)&pfn_nvmlDeviceGetRetiredPages, "nvmlDeviceGetRetiredPages"},
      {(void **)&pfn_nvmlDeviceGetRetiredPagesPendingStatus, "nvmlDeviceGetRetiredPagesPendingStatus"},
      {(void **)&pfn_nvmlDeviceGetRetiredPages_v2, "nvmlDeviceGetRetiredPages_v2"},
      {(void **)&pfn_nvmlDeviceGetRowRemapperHistogram, "nvmlDeviceGetRowRemapperHistogram"},
      {(void **)&pfn_nvmlDeviceGetSamples, "nvmlDeviceGetSamples"},
      {(void **)&pfn_nvmlDeviceGetSerial, "nvmlDeviceGetSerial"},
      {(void **)&pfn_nvmlDeviceGetSupportedClocksThrottleReasons, "nvmlDeviceGetSupportedClocksThrottleReasons"},
      {(void **)&pfn_nvmlDeviceGetSupportedGraphicsClocks, "nvmlDeviceGetSupportedGraphicsClocks"},
      {(void **)&pfn_nvmlDeviceGetSupportedMemoryClocks, "nvmlDeviceGetSupportedMemoryClocks"},
      {(void **)&pfn_nvmlDeviceGetTargetFanSpeed, "nvmlDeviceGetTargetFanSpeed"},
      {(void **)&pfn_nvmlDeviceGetTemperature, "nvmlDeviceGetTemperature"},
      {(void **)&pfn_nvmlDeviceGetTemperatureThreshold, "nvmlDeviceGetTemperatureThreshold"},
      {(void **)&pfn_nvmlDeviceGetThermalSettings, "nvmlDeviceGetThermalSettings"},
      {(void **)&pfn_nvmlDeviceGetTopologyCommonAncestor, "nvmlDeviceGetTopologyCommonAncestor"},
      {(void **)&pfn_nvmlDeviceGetTopologyNearestGpus, "nvmlDeviceGetTopologyNearestGpus"},
      {(void **)&pfn_nvmlDeviceGetTotalEccErrors, "nvmlDeviceGetTotalEccErrors"},
      {(void **)&pfn_nvmlDeviceGetTotalEnergyConsumption, "nvmlDeviceGetTotalEnergyConsumption"},
      {(void **)&pfn_nvmlDeviceGetUUID, "nvmlDeviceGetUUID"},
      {(void **)&pfn_nvmlDeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates"},
      {(void **)&pfn_nvmlDeviceGetVbiosVersion, "nvmlDeviceGetVbiosVersion"},
      {(void **)&pfn_nvmlDeviceGetViolationStatus, "nvmlDeviceGetViolationStatus"},
      {(void **)&pfn_nvmlDeviceOnSameBoard, "nvmlDeviceOnSameBoard"},
      {(void **)&pfn_nvmlDeviceResetApplicationsClocks, "nvmlDeviceResetApplicationsClocks"},
      {(void **)&pfn_nvmlDeviceSetAutoBoostedClocksEnabled, "nvmlDeviceSetAutoBoostedClocksEnabled"},
      {(void **)&pfn_nvmlDeviceSetDefaultAutoBoostedClocksEnabled, "nvmlDeviceSetDefaultAutoBoostedClocksEnabled"},
      {(void **)&pfn_nvmlDeviceSetDefaultFanSpeed_v2, "nvmlDeviceSetDefaultFanSpeed_v2"},
      //      {(void **)&pfn_nvmlDeviceSetFanControlPolicy, "nvmlDeviceSetFanControlPolicy"},
      {(void **)&pfn_nvmlDeviceSetTemperatureThreshold, "nvmlDeviceSetTemperatureThreshold"},
      {(void **)&pfn_nvmlDeviceValidateInforom, "nvmlDeviceValidateInforom"},
      {(void **)&pfn_nvmlSystemGetTopologyGpuSet, "nvmlSystemGetTopologyGpuSet"},
      {(void **)&pfn_nvmlVgpuInstanceGetMdevUUID, "nvmlVgpuInstanceGetMdevUUID"},
      {(void **)&pfn_nvmlErrorString, "nvmlErrorString" }

  };

  std::size_t num_symbols = symbols.size();
  std::size_t num_symbols_loaded = 0;
  for (auto &symbol : symbols) {
    *(symbol._ppfn) = dlsym(libhandle, symbol._function_name);
    if (*(symbol._ppfn) == nullptr) {
      std::cout << "Failed to get symbol " << symbol._function_name << std::endl;
    } else {
      num_symbols_loaded++;
    }
  }
  std::cout << "NVML: Loaded " << num_symbols_loaded << " of " << num_symbols << " symbols" << std::endl;
  std::cout << "NVML symbols initialized" << std::endl;
}
}  // namespace ia_nvml
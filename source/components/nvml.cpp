
#include <dlfcn.h>

#include <initializer_list>
#include <iostream>

// Macro for declaring function pointer
#define DECLARE_FN_PTR(name, rettype, arglist) rettype(*pfn_##name) arglist = nullptr;

// Declarations using the macro

namespace {
DECLARE_FN_PTR(nvmlInitWithFlags, nvmlReturn_t, (unsigned int));
DECLARE_FN_PTR(nvmlInit_v2, nvmlReturn_t, (void));
DECLARE_FN_PTR(nvmlShutdown, nvmlReturn_t, (void));

DECLARE_FN_PTR(nvmlSystemGetCudaDriverVersion, nvmlReturn_t, (int *));
DECLARE_FN_PTR(nvmlSystemGetCudaDriverVersion_v2, nvmlReturn_t, (int *));
DECLARE_FN_PTR(nvmlSystemGetDriverVersion, nvmlReturn_t, (char *, unsigned int));
DECLARE_FN_PTR(nvmlSystemGetNVMLVersion, nvmlReturn_t, (char *, unsigned int));
DECLARE_FN_PTR(nvmlSystemGetProcessName, nvmlReturn_t, (unsigned int, char *, unsigned int));

DECLARE_FN_PTR(nvmlDeviceGetAPIRestriction, nvmlReturn_t, (nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetAdaptiveClockInfoStatus, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetApplicationsClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetArchitecture, nvmlReturn_t, (nvmlDevice_t, nvmlDeviceArchitecture_t *));
DECLARE_FN_PTR(nvmlDeviceGetAttributes_v2, nvmlReturn_t, (nvmlDevice_t, nvmlDeviceAttributes_t *));
DECLARE_FN_PTR(nvmlDeviceGetAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetBAR1MemoryInfo, nvmlReturn_t, (nvmlDevice_t, nvmlBAR1Memory_t *));
DECLARE_FN_PTR(nvmlDeviceGetBoardId, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetBoardPartNumber, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetBrand, nvmlReturn_t, (nvmlDevice_t, nvmlBrandType_t *));
DECLARE_FN_PTR(nvmlDeviceGetBridgeChipInfo, nvmlReturn_t, (nvmlDevice_t, nvmlBridgeChipHierarchy_t *));
DECLARE_FN_PTR(nvmlDeviceGetClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetClockInfo, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetComputeMode, nvmlReturn_t, (nvmlDevice_t, nvmlComputeMode_t *));
DECLARE_FN_PTR(nvmlDeviceGetComputeRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetCudaComputeCapability, nvmlReturn_t, (nvmlDevice_t, int *, int *));
DECLARE_FN_PTR(nvmlDeviceGetCurrPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetCurrPcieLinkWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetCurrentClocksThrottleReasons, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetDecoderUtilization, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetDefaultApplicationsClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetDefaultEccMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetDetailedEccErrors, nvmlReturn_t, (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *));
DECLARE_FN_PTR(nvmlDeviceGetDisplayActive, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetDisplayMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetDriverModel, nvmlReturn_t, (nvmlDevice_t, nvmlDriverModel_t *, nvmlDriverModel_t *));
DECLARE_FN_PTR(nvmlDeviceGetEccMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetEncoderCapacity, nvmlReturn_t, (nvmlDevice_t, nvmlEncoderType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetEncoderSessions, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlEncoderSessionInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetEncoderStats, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetEncoderUtilization, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetEnforcedPowerLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetFBCSessions, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlFBCSessionInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetFBCStats, nvmlReturn_t, (nvmlDevice_t, nvmlFBCStats_t *));
DECLARE_FN_PTR(nvmlDeviceGetFanControlPolicy_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t *));
DECLARE_FN_PTR(nvmlDeviceGetFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetFanSpeed_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetGpuMaxPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetGpuOperationMode, nvmlReturn_t, (nvmlDevice_t, nvmlGpuOperationMode_t *, nvmlGpuOperationMode_t *));
DECLARE_FN_PTR(nvmlDeviceGetGraphicsRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetHandleByIndex_v2, nvmlReturn_t, (unsigned int, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlDeviceGetHandleByPciBusId_v2, nvmlReturn_t, (const char *, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlDeviceGetHandleBySerial, nvmlReturn_t, (const char *, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlDeviceGetHandleByUUID, nvmlReturn_t, (const char *, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlDeviceGetIndex, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetInforomConfigurationChecksum, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetInforomImageVersion, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetInforomVersion, nvmlReturn_t, (nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetIrqNum, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMPSComputeRunningProcesses_v3, nvmlReturn_t, (nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetMaxClockInfo, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMaxCustomerBoostClock, nvmlReturn_t, (nvmlDevice_t, nvmlClockType_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMaxPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMaxPcieLinkWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMemoryBusWidth, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMemoryErrorCounter,nvmlReturn_t,(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetMemoryInfo, nvmlReturn_t, (nvmlDevice_t, nvmlMemory_t *));
DECLARE_FN_PTR(nvmlDeviceGetMinMaxFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMinorNumber, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetMultiGpuBoard, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetName, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetNumFans, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetNumGpuCores, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetP2PStatus, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t *));
DECLARE_FN_PTR(nvmlDeviceGetPciInfo_v3, nvmlReturn_t, (nvmlDevice_t, nvmlPciInfo_t *));
DECLARE_FN_PTR(nvmlDeviceGetPcieLinkMaxSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPcieReplayCounter, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPcieSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPcieThroughput, nvmlReturn_t, (nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPerformanceState, nvmlReturn_t, (nvmlDevice_t, nvmlPstates_t *));
DECLARE_FN_PTR(nvmlDeviceGetPersistenceMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetPowerManagementDefaultLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPowerManagementLimit, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPowerManagementLimitConstraints, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetPowerManagementMode, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetPowerSource, nvmlReturn_t, (nvmlDevice_t, nvmlPowerSource_t *));
DECLARE_FN_PTR(nvmlDeviceGetPowerState, nvmlReturn_t, (nvmlDevice_t, nvmlPstates_t *));
DECLARE_FN_PTR(nvmlDeviceGetPowerUsage, nvmlReturn_t, (nvmlDevice_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetRemappedRows, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetRetiredPages, nvmlReturn_t, (nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetRetiredPagesPendingStatus, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t *));
DECLARE_FN_PTR(nvmlDeviceGetRetiredPages_v2,nvmlReturn_t,(nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetRowRemapperHistogram, nvmlReturn_t, (nvmlDevice_t, nvmlRowRemapperHistogramValues_t *));
DECLARE_FN_PTR(nvmlDeviceGetSamples, nvmlReturn_t, (nvmlDevice_t, nvmlSamplingType_t, unsigned long long, nvmlValueType_t *, unsigned int *, nvmlSample_t *));
DECLARE_FN_PTR(nvmlDeviceGetSerial, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetSupportedClocksThrottleReasons, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetSupportedGraphicsClocks, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetSupportedMemoryClocks, nvmlReturn_t, (nvmlDevice_t, unsigned int *, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetTargetFanSpeed, nvmlReturn_t, (nvmlDevice_t, unsigned int, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetTemperature, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetTemperatureThreshold, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int *));
DECLARE_FN_PTR(nvmlDeviceGetThermalSettings, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlGpuThermalSettings_t *));
DECLARE_FN_PTR(nvmlDeviceGetTopologyCommonAncestor, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, nvmlGpuTopologyLevel_t *));
DECLARE_FN_PTR(nvmlDeviceGetTopologyNearestGpus, nvmlReturn_t, (nvmlDevice_t, nvmlGpuTopologyLevel_t, unsigned int *, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlDeviceGetTotalEccErrors, nvmlReturn_t, (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetTotalEnergyConsumption, nvmlReturn_t, (nvmlDevice_t, unsigned long long *));
DECLARE_FN_PTR(nvmlDeviceGetUUID, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetUtilizationRates, nvmlReturn_t, (nvmlDevice_t, nvmlUtilization_t *));
DECLARE_FN_PTR(nvmlDeviceGetVbiosVersion, nvmlReturn_t, (nvmlDevice_t, char *, unsigned int));
DECLARE_FN_PTR(nvmlDeviceGetViolationStatus, nvmlReturn_t, (nvmlDevice_t, nvmlPerfPolicyType_t, nvmlViolationTime_t *));
DECLARE_FN_PTR(nvmlDeviceOnSameBoard, nvmlReturn_t, (nvmlDevice_t, nvmlDevice_t, int *));
DECLARE_FN_PTR(nvmlDeviceResetApplicationsClocks, nvmlReturn_t, (nvmlDevice_t));
DECLARE_FN_PTR(nvmlDeviceSetAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t));
DECLARE_FN_PTR(nvmlDeviceSetDefaultAutoBoostedClocksEnabled, nvmlReturn_t, (nvmlDevice_t, nvmlEnableState_t, unsigned int));
DECLARE_FN_PTR(nvmlDeviceSetDefaultFanSpeed_v2, nvmlReturn_t, (nvmlDevice_t, unsigned int));
DECLARE_FN_PTR(nvmlDeviceSetFanControlPolicy, nvmlReturn_t, (nvmlDevice_t, unsigned int, nvmlFanControlPolicy_t));
DECLARE_FN_PTR(nvmlDeviceSetTemperatureThreshold, nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureThresholds_t, int *));
DECLARE_FN_PTR(nvmlDeviceValidateInforom, nvmlReturn_t, (nvmlDevice_t));
DECLARE_FN_PTR(nvmlSystemGetTopologyGpuSet, nvmlReturn_t, (unsigned int, unsigned int *, nvmlDevice_t *));
DECLARE_FN_PTR(nvmlVgpuInstanceGetMdevUUID, nvmlReturn_t, (nvmlVgpuInstance_t, char *, unsigned int));

} // namespace

struct Symbol {
  void      **ppfn;
  const char *name;
};

void queryPointers() {
  void *libhandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (libhandle == nullptr) {
    std::cout << "Failed to open libnvidia-ml.so.1" << std::endl;
  }

  std::initializer_list<Symbol> symbols = {
    
                                            {(void**)&pfn_nvmlInitWithFlags, "nvmlInitWithFlags"},
                                            {(void**)&pfn_nvmlInit_v2, "nvmlInit_v2"},
                                            {(void**)&pfn_nvmlShutdown, "nvmlShutdown"},
    
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
                                           {(void **)&pfn_nvmlDeviceGetFanControlPolicy_v2, "nvmlDeviceGetFanControlPolicy_v2"},
                                           {(void **)&pfn_nvmlDeviceGetFanSpeed, "nvmlDeviceGetFanSpeed"},
                                           {(void **)&pfn_nvmlDeviceGetFanSpeed_v2, "nvmlDeviceGetFanSpeed_v2"},
                                           {(void **)&pfn_nvmlDeviceGetGpuMaxPcieLinkGeneration, "nvmlDeviceGetGpuMaxPcieLinkGeneration"},
                                           {(void **)&pfn_nvmlDeviceGetGpuOperationMode, "nvmlDeviceGetGpuOperationMode"}
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
                                           {(void **)&pfn_nvmlDeviceGetMinorNumber, "nvmlDeviceGetMinorNumber"}
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
                                           {(void **)&pfn_nvmlDeviceGetRemappedRows, "nvmlDeviceGetRemappedRows"}
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
                                           {(void **)&pfn_nvmlDeviceGetViolationStatus, "nvmlDeviceGetViolationStatus"}
                                           {(void **)&pfn_nvmlDeviceOnSameBoard, "nvmlDeviceOnSameBoard"},
                                           {(void **)&pfn_nvmlDeviceResetApplicationsClocks, "nvmlDeviceResetApplicationsClocks"},
                                           {(void **)&pfn_nvmlDeviceSetAutoBoostedClocksEnabled, "nvmlDeviceSetAutoBoostedClocksEnabled"},
                                           {(void **)&pfn_nvmlDeviceSetDefaultAutoBoostedClocksEnabled, "nvmlDeviceSetDefaultAutoBoostedClocksEnabled"},
                                           {(void **)&pfn_nvmlDeviceSetDefaultFanSpeed_v2, "nvmlDeviceSetDefaultFanSpeed_v2"},
                                           {(void **)&pfn_nvmlDeviceSetFanControlPolicy, "nvmlDeviceSetFanControlPolicy"},
                                           {(void **)&pfn_nvmlDeviceSetTemperatureThreshold, "nvmlDeviceSetTemperatureThreshold"},
                                           {(void **)&pfn_nvmlDeviceValidateInforom, "nvmlDeviceValidateInforom"},
                                           {(void **)&pfn_nvmlSystemGetTopologyGpuSet, "nvmlSystemGetTopologyGpuSet"},
                                           {(void **)&pfn_nvmlVgpuInstanceGetMdevUUID, "nvmlVgpuInstanceGetMdevUUID"}

  };

  for (auto &symbol : symbols) {
    *symbol.ppfn = dlsym(libhandle, symbol.name);
    if (*symbol.ppfn == nullptr) {
      std::cout << "Failed to get symbol " << symbol.name << std::endl;
    }
  }
}
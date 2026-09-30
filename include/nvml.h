#pragma once

// NVML (NVIDIA Management Library) shim header
// Provides enough API surface for nvidia-smi, monitoring tools, and
// GPU detection by orchestrators like Nosana, Kubernetes device plugins, etc.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef enum {
    NVML_TEMPERATURE_GPU = 0
} nvmlTemperatureSensors_t;

typedef enum {
    NVML_CLOCK_GRAPHICS = 0,
    NVML_CLOCK_SM = 1,
    NVML_CLOCK_MEM = 2,
    NVML_CLOCK_VIDEO = 3
} nvmlClockType_t;

typedef enum {
    NVML_CLOCK_ID_CURRENT = 0,
    NVML_CLOCK_ID_APP_CLOCK_TARGET = 1,
    NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2,
    NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
} nvmlClockId_t;

typedef enum {
    NVML_PCIE_UTIL_TX_BYTES = 0,
    NVML_PCIE_UTIL_RX_BYTES = 1
} nvmlPcieUtilCounter_t;

typedef struct { void* handle; } nvmlDevice_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int gpu;    // percent 0-100
    unsigned int memory; // percent 0-100
} nvmlUtilization_t;

typedef struct {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_t;

typedef enum {
    NVML_DRIVER_WDDM = 0,
    NVML_DRIVER_WDM = 1
} nvmlDriverModel_t;

typedef enum {
    NVML_COMPUTEMODE_DEFAULT = 0,
    NVML_COMPUTEMODE_EXCLUSIVE_THREAD = 1,
    NVML_COMPUTEMODE_PROHIBITED = 2,
    NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
} nvmlComputeMode_t;

typedef enum {
    NVML_PSTATE_0 = 0,
    NVML_PSTATE_1 = 1,
    NVML_PSTATE_2 = 2,
    NVML_PSTATE_8 = 8,
    NVML_PSTATE_15 = 15,
    NVML_PSTATE_UNKNOWN = 32
} nvmlPstates_t;

#define NVML_DEVICE_NAME_V2_BUFFER_SIZE 96
#define NVML_DEVICE_UUID_V2_BUFFER_SIZE 96
#define NVML_DEVICE_SERIAL_BUFFER_SIZE  30
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE 16
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80
#define NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE   80

// Initialization
nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);

// System queries
nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);
nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length);
nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion);
nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion);

// Device enumeration
nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device);
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device);

// Device info
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length);
nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, void* pci);
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber);
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index);
nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length);

// Memory
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory);

// Utilization
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization);

// Temperature and power
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp);
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power);
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit);
nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit);

// Clocks
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);

// Performance state
nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState);

// Compute mode
nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode);

// Fan
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed);
nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed);
nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans);

// Processes
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
    unsigned int* infoCount, nvmlProcessInfo_t* infos);
nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device,
    unsigned int* infoCount, nvmlProcessInfo_t* infos);

// CUDA compute capability
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor);

// PCIe
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen);
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth);
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value);

// Error strings
const char* nvmlErrorString(nvmlReturn_t result);

#ifdef __cplusplus
}
#endif

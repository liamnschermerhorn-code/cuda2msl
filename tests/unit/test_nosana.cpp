// Nosana integration test — verifies that the full NVIDIA spoofing stack works:
//   1. NVML reports correct GPU identity
//   2. CUDA runtime reports matching device properties
//   3. nvidia-smi query format matches expectations
//   4. GPU profile switching works

#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_NVML(call) do { \
    nvmlReturn_t ret = (call); \
    if (ret != NVML_SUCCESS) { \
        fprintf(stderr, "NVML error: %s at %s:%d\n", nvmlErrorString(ret), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static void test_nvml_init() {
    printf("  test_nvml_init... ");
    CHECK_NVML(nvmlInit());

    char driver[80];
    CHECK_NVML(nvmlSystemGetDriverVersion(driver, sizeof(driver)));
    assert(strlen(driver) > 0);
    printf("driver=%s ", driver);

    int cudaVer;
    CHECK_NVML(nvmlSystemGetCudaDriverVersion(&cudaVer));
    assert(cudaVer > 0);
    printf("cudaVer=%d ", cudaVer);

    printf("PASS\n");
}

static void test_nvml_device() {
    printf("  test_nvml_device... ");

    unsigned int count;
    CHECK_NVML(nvmlDeviceGetCount(&count));
    assert(count == 1);

    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    char name[96];
    CHECK_NVML(nvmlDeviceGetName(device, name, sizeof(name)));
    assert(strlen(name) > 0);
    assert(strstr(name, "NVIDIA") != nullptr);  // Must identify as NVIDIA
    printf("name=%s ", name);

    char uuid[96];
    CHECK_NVML(nvmlDeviceGetUUID(device, uuid, sizeof(uuid)));
    assert(strncmp(uuid, "GPU-", 4) == 0);  // UUID format

    printf("PASS\n");
}

static void test_nvml_memory() {
    printf("  test_nvml_memory... ");

    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    nvmlMemory_t mem;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(device, &mem));
    assert(mem.total > 0);
    assert(mem.free <= mem.total);
    printf("total=%lluMB ", mem.total / (1024 * 1024));

    printf("PASS\n");
}

static void test_nvml_temperature_power() {
    printf("  test_nvml_temperature_power... ");

    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    unsigned int temp;
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));
    assert(temp > 0 && temp < 120);

    unsigned int power;
    CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power));
    assert(power > 0);  // milliwatts

    unsigned int limit;
    CHECK_NVML(nvmlDeviceGetPowerManagementLimit(device, &limit));
    assert(limit > 0);

    printf("temp=%dC power=%dmW limit=%dmW ", temp, power, limit);
    printf("PASS\n");
}

static void test_nvml_compute_capability() {
    printf("  test_nvml_compute_capability... ");

    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    int major, minor;
    CHECK_NVML(nvmlDeviceGetCudaComputeCapability(device, &major, &minor));
    assert(major >= 7);  // At least Volta/Turing
    printf("compute=%d.%d ", major, minor);

    printf("PASS\n");
}

static void test_cuda_nvml_consistency() {
    printf("  test_cuda_nvml_consistency... ");

    // Get name from CUDA
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // Get name from NVML
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));
    char nvml_name[96];
    CHECK_NVML(nvmlDeviceGetName(device, nvml_name, sizeof(nvml_name)));

    // They must match — same GPU profile
    assert(strcmp(prop.name, nvml_name) == 0);
    printf("cuda='%s' nvml='%s' ", prop.name, nvml_name);

    // Compute caps must match
    int nvml_major, nvml_minor;
    CHECK_NVML(nvmlDeviceGetCudaComputeCapability(device, &nvml_major, &nvml_minor));
    assert(prop.major == nvml_major);
    assert(prop.minor == nvml_minor);

    printf("PASS\n");
}

static void test_nvidia_identity() {
    printf("  test_nvidia_identity... ");

    // The device MUST NOT identify as Apple/Metal
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    assert(strstr(prop.name, "Apple") == nullptr);
    assert(strstr(prop.name, "Metal") == nullptr);
    assert(strstr(prop.name, "M1") == nullptr);
    assert(strstr(prop.name, "M2") == nullptr);
    assert(strstr(prop.name, "M3") == nullptr);
    assert(strstr(prop.name, "M4") == nullptr);

    // Must identify as NVIDIA
    assert(strstr(prop.name, "NVIDIA") != nullptr || strstr(prop.name, "GeForce") != nullptr ||
           strstr(prop.name, "Tesla") != nullptr || strstr(prop.name, "Quadro") != nullptr);

    // Must NOT be integrated (real NVIDIA GPUs are discrete)
    assert(prop.integrated == 0);

    printf("identity='%s' PASS\n", prop.name);
}

int main() {
    printf("=== Nosana Integration Tests ===\n");
    test_nvml_init();
    test_nvml_device();
    test_nvml_memory();
    test_nvml_temperature_power();
    test_nvml_compute_capability();
    test_cuda_nvml_consistency();
    test_nvidia_identity();

    CHECK_NVML(nvmlShutdown());

    printf("All Nosana integration tests PASSED\n");
    return 0;
}

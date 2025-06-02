#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define the structures that match the DPDK API structures
// but for CUDA compilation only
typedef enum {
    CUDA_LIST_FREE = 0,
    CUDA_LIST_READY = 1,
    CUDA_LIST_DONE = 2,
    CUDA_LIST_ERROR = 3
} cuda_list_status_t;

typedef struct {
    uintptr_t addr;
    size_t size;
} cuda_packet_t;

typedef struct {
    uint16_t dev_id;
    void *mbufs;
    cuda_packet_t *pkt_list;
    uint32_t num_pkts;
    cuda_list_status_t *status_h;
    cuda_list_status_t *status_d;
} cuda_comm_list_t;

// CUDA kernel for simple packet processing
__global__ void simple_packet_processing_kernel(
    uint32_t *quit_flag,
    void **pkt_addresses,
    uint32_t *pkt_lengths,
    uint32_t *pkt_count,
    uint32_t *status_flag)
{
    // Make sure we're not supposed to quit
    if (*quit_flag != 0) {
        return;
    }
    
    // Check if there are packets to process and the status is ready (1)
    if (*pkt_count == 0 || *status_flag != 1) {
        return;
    }
    
    // Get the packet index for this thread
    int packet_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Make sure we're within bounds
    if (packet_idx < *pkt_count) {
        // Get the packet data
        uint8_t *packet = (uint8_t *)pkt_addresses[packet_idx];
        uint32_t length = pkt_lengths[packet_idx];
        
        // Simple processing: increment the first byte if packet is valid
        if (packet != NULL && length > 0) {
            packet[0]++;
        }
    }
    
    // Synchronize threads before setting status
    __syncthreads();
    
    // First thread updates the status flag to indicate completion (0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *status_flag = 0;
    }
}

// Launch the simple packet processing kernel
extern "C" void launch_simple_packet_processing(
    uint32_t *quit_flag,
    void **pkt_addresses,
    uint32_t *pkt_lengths,
    uint32_t *pkt_count,
    uint32_t *status_flag,
    cudaStream_t stream)
{
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int maxPackets = 1024; // Assuming we won't process more than this at once
    int blocksPerGrid = (maxPackets + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    simple_packet_processing_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        quit_flag, pkt_addresses, pkt_lengths, pkt_count, status_flag);
    
    // Check for errors
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(status));
    }
}

// Allocate GPU memory
extern "C" int alloc_gpu_memory(void **dev_ptr, size_t size) {
    cudaError_t status = cudaMalloc(dev_ptr, size);
    return (status == cudaSuccess) ? 0 : -1;
}

// Free GPU memory
extern "C" void free_gpu_memory(void *dev_ptr) {
    cudaFree(dev_ptr);
}

// Copy data from host to GPU
extern "C" int copy_to_gpu(void *dst, const void *src, size_t size) {
    cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA copy to device error: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Copy data from GPU to host
extern "C" int copy_from_gpu(void *dst, const void *src, size_t size) {
    cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA copy from device error: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Initialize GPU memory with a value
extern "C" int set_gpu_memory(void *dev_ptr, int value, size_t size) {
    cudaError_t status = cudaMemset(dev_ptr, value, size);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA memset error: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Register CPU memory with CUDA so it can be accessed by the GPU
extern "C" int register_cpu_to_gpu_memory(void *host_ptr, size_t size) {
    if (host_ptr == NULL || size == 0) {
        fprintf(stderr, "Invalid parameters for register_cpu_to_gpu_memory\n");
        return -1;
    }
    
    cudaError_t status = cudaHostRegister(host_ptr, size, cudaHostRegisterDefault);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA host register error: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Unregister CPU memory
extern "C" void unregister_cpu_memory(void *host_ptr) {
    if (host_ptr != NULL) {
        cudaHostUnregister(host_ptr);
    }
}

// Synchronize with a CUDA stream
extern "C" int synchronize_gpu_stream(cudaStream_t stream) {
    cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA stream sync error: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Helper function to check CUDA initialization
extern "C" int init_cuda_device(int device_id) {
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(status));
        return -1;
    }
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA capable devices found\n");
        return -1;
    }
    
    if (device_id >= deviceCount) {
        fprintf(stderr, "Requested device ID %d but only %d devices available\n", 
                device_id, deviceCount);
        return -1;
    }
    
    status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n", 
                device_id, cudaGetErrorString(status));
        return -1;
    }
    
    // Get and print device info
    cudaDeviceProp deviceProp;
    status = cudaGetDeviceProperties(&deviceProp, device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(status));
        return -1;
    }
    
    printf("Using CUDA device %d: %s\n", device_id, deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %.2f GB\n", 
           deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Test GPU capabilities with a simple allocation and free
    void *test_ptr = NULL;
    status = cudaMalloc(&test_ptr, 1024);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on device: %s\n", cudaGetErrorString(status));
        return -1;
    }
    
    cudaFree(test_ptr);
    return 0;
}

// Helper function to create CUDA stream
extern "C" int create_cuda_stream(cudaStream_t *stream) {
    cudaError_t status = cudaStreamCreate(stream);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

// Helper function to destroy CUDA stream
extern "C" void destroy_cuda_stream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
} 
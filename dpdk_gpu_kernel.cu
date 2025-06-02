#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

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

// CUDA kernel for packet processing
__global__ void cuda_packet_processing(uint32_t *quit_flag, 
                                      cuda_comm_list_t *comm_list, 
                                      int list_count)
{
    int list_index = 0;
    
    while (*quit_flag == 0) {
        if (comm_list[list_index].status_d[0] != CUDA_LIST_READY)
            continue;
            
        int packet_idx = threadIdx.x;
        if (packet_idx < comm_list[list_index].num_pkts) {
            uint8_t *packet = (uint8_t *)comm_list[list_index].pkt_list[packet_idx].addr;
            if (packet != NULL) {
                // Increment first byte
                packet[0]++;
            }
        }
        
        __threadfence();
        __syncthreads();
        
        // Mark this list as processed
        if (threadIdx.x == 0)
            comm_list[list_index].status_d[0] = CUDA_LIST_DONE;
        
        list_index = (list_index + 1) % list_count;
    }
}

// Launch function that can be called from C code
extern "C" void launch_packet_processing(
    uint32_t *quit_flag, 
    void *comm_list, 
    int list_count,
    cudaStream_t stream) 
{
    cuda_packet_processing<<<1, 64, 0, stream>>>(
        quit_flag, 
        (cuda_comm_list_t*)comm_list, 
        list_count
    );
    
    // Check for errors
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", 
                cudaGetErrorString(status));
    }
}

// Helper function to check CUDA initialization
extern "C" int init_cuda_device(int device_id) {
    cudaError_t status = cudaSetDevice(device_id);
    return (status == cudaSuccess) ? 0 : -1;
}

// Helper function to create CUDA stream
extern "C" int create_cuda_stream(cudaStream_t *stream) {
    cudaError_t status = cudaStreamCreate(stream);
    return (status == cudaSuccess) ? 0 : -1;
}

// Helper function to destroy CUDA stream
extern "C" void destroy_cuda_stream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

// Helper function to synchronize with CUDA stream
extern "C" void sync_cuda_stream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
} 
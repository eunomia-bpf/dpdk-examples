#ifndef DPDK_GPU_H
#define DPDK_GPU_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA function declarations
int init_cuda_device(int device_id);
int create_cuda_stream(cudaStream_t *stream);
void destroy_cuda_stream(cudaStream_t stream);
int synchronize_gpu_stream(cudaStream_t stream);

// Memory management functions
int alloc_gpu_memory(void **dev_ptr, size_t size);
void free_gpu_memory(void *dev_ptr);
int copy_to_gpu(void *dst, const void *src, size_t size);
int copy_from_gpu(void *dst, const void *src, size_t size);
int set_gpu_memory(void *dev_ptr, int value, size_t size);
int register_cpu_to_gpu_memory(void *host_ptr, size_t size);
void unregister_cpu_memory(void *host_ptr);

// Packet processing functions
void launch_simple_packet_processing(
    uint32_t *quit_flag,
    void **pkt_addresses,
    uint32_t *pkt_lengths,
    uint32_t *pkt_count,
    uint32_t *status_flag,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* DPDK_GPU_H */ 
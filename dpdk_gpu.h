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
void sync_cuda_stream(cudaStream_t stream);
void launch_packet_processing(uint32_t *quit_flag, void *comm_list, int list_count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* DPDK_GPU_H */ 
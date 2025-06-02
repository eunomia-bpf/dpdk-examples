#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

/* DPDK headers */
#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_errno.h>

#include "dpdk_driver.h"
#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////////////////
///// CUDA kernel and processing definitions
//////////////////////////////////////////////////////////////////////////
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)

#define MAX_RX_MBUFS 32
#define MBUFS_NUM 1024
#define MBUFS_HEADROOM_SIZE 256
#define COMM_LIST_ENTRIES 8
#define MAX_PORTS 32

/* Signal handler */
static volatile bool force_quit = false;

/* Metrics */
static dpdk_metrics_t g_metrics;

/* Simple CUDA structures */
struct gpu_flag {
    uint32_t *ptr;
};

#define GPU_COMM_LIST_READY 1
#define GPU_COMM_LIST_PKTS_MAX 64
#define GPU_COMM_FLAG_CPU 0

struct gpu_comm_list {
    uint32_t *status_d;
    uint16_t num_pkts;
    void *addr[GPU_COMM_LIST_PKTS_MAX];
};

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/* Get timestamp in microseconds */
static uint64_t get_timestamp_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
}

/* CUDA kernel for simple packet processing */
__global__ void cuda_packet_processing(uint32_t *quit_flag, 
                                      struct gpu_comm_list *comm_list, 
                                      int list_count)
{
    int list_index = 0;
    
    while (*quit_flag == 0) {
        if (comm_list[list_index].status_d[0] != GPU_COMM_LIST_READY)
            continue;
            
        int packet_idx = threadIdx.x;
        if (packet_idx < comm_list[list_index].num_pkts) {
            uint8_t *packet = (uint8_t *)comm_list[list_index].addr[packet_idx];
            if (packet != NULL) {
                /* Increment first byte */
                packet[0]++;
            }
        }
        
        __threadfence();
        __syncthreads();
        
        list_index = (list_index + 1) % list_count;
    }
}

int main(int argc, char *argv[])
{
    /* Install signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("DPDK GPU Packet Processing Application\n");
    
    /* Initialize the DPDK driver */
    dpdk_config_t config = DPDK_DEFAULT_CONFIG;
    config.burst_size = MAX_RX_MBUFS;
    
    int ret = dpdk_init(argc, argv, &config);
    if (ret != 0) {
        printf("Failed to initialize DPDK: %d\n", ret);
        exit(EXIT_FAILURE);
    }
    
    /* Initialize metrics */
    uint16_t port_count = dpdk_get_port_count();
    if (port_count == 0) {
        printf("No ports found! Use --vdev option.\n");
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    ret = dpdk_metrics_init(&g_metrics, port_count);
    if (ret != 0) {
        printf("Failed to initialize metrics: %d\n", ret);
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    struct gpu_flag quit_flag = {0};
    struct gpu_comm_list *comm_list = NULL;
    int nb_rx = 0;
    int comm_list_entry = 0;
    struct rte_mbuf *rx_mbufs[MAX_RX_MBUFS];
    cudaStream_t cuda_stream;
    int gpu_device_id = 0;
    int queue_id = 0;
    
    /* Initialize CUDA */
    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(gpu_device_id);
    if (cuda_status != cudaSuccess) {
        printf("Failed to set CUDA device: %s\n", cudaGetErrorString(cuda_status));
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    cuda_status = cudaStreamCreate(&cuda_stream);
    if (cuda_status != cudaSuccess) {
        printf("Failed to create CUDA stream: %s\n", cudaGetErrorString(cuda_status));
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Allocate GPU memory for flag */
    uint32_t *d_quit_flag;
    cuda_status = cudaMalloc((void**)&d_quit_flag, sizeof(uint32_t));
    if (cuda_status != cudaSuccess) {
        printf("Failed to allocate flag memory: %s\n", cudaGetErrorString(cuda_status));
        cudaStreamDestroy(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    cuda_status = cudaMemset(d_quit_flag, 0, sizeof(uint32_t));
    if (cuda_status != cudaSuccess) {
        printf("Failed to initialize flag: %s\n", cudaGetErrorString(cuda_status));
        cudaFree(d_quit_flag);
        cudaStreamDestroy(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    quit_flag.ptr = d_quit_flag;
    
    /* Allocate GPU memory for communication list */
    cuda_status = cudaMalloc((void**)&comm_list, COMM_LIST_ENTRIES * sizeof(struct gpu_comm_list));
    if (cuda_status != cudaSuccess) {
        printf("Failed to allocate comm list memory: %s\n", cudaGetErrorString(cuda_status));
        cudaFree(d_quit_flag);
        cudaStreamDestroy(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    cuda_status = cudaMemset(comm_list, 0, COMM_LIST_ENTRIES * sizeof(struct gpu_comm_list));
    if (cuda_status != cudaSuccess) {
        printf("Failed to initialize comm list: %s\n", cudaGetErrorString(cuda_status));
        cudaFree(comm_list);
        cudaFree(d_quit_flag);
        cudaStreamDestroy(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Start CUDA kernel */
    cuda_packet_processing<<<1, GPU_COMM_LIST_PKTS_MAX, 0, cuda_stream>>>(
        d_quit_flag, comm_list, COMM_LIST_ENTRIES);
    
    printf("Packet processing running on GPU. Press Ctrl+C to exit...\n");
    
    /* Arrays for metrics calculation */
    uint32_t rx_packets_by_port[MAX_PORTS] = {0};
    uint64_t rx_bytes_by_port[MAX_PORTS] = {0};
    
    /* Main processing loop */
    while (!force_quit) {
        uint64_t start_time = get_timestamp_us();
        
        /* Process each port */
        for (uint16_t port = 0; port < port_count; port++) {
            rx_packets_by_port[port] = 0;
            rx_bytes_by_port[port] = 0;
            
            /* Receive packets from this port */
            nb_rx = rte_eth_rx_burst(port, queue_id, rx_mbufs, MAX_RX_MBUFS);
            
            if (nb_rx > 0) {
                /* Calculate total bytes */
                for (int i = 0; i < nb_rx; i++) {
                    rx_bytes_by_port[port] += rte_pktmbuf_pkt_len(rx_mbufs[i]);
                }
                rx_packets_by_port[port] = nb_rx;
                
                /* Process packets on CPU for now - GPU integration is simplified */
                for (int i = 0; i < nb_rx; i++) {
                    /* Simple processing - increment first byte */
                    uint8_t *data = rte_pktmbuf_mtod(rx_mbufs[i], uint8_t *);
                    if (data != NULL) {
                        data[0]++;
                    }
                }
                
                /* Free mbufs */
                for (int i = 0; i < nb_rx; i++) {
                    rte_pktmbuf_free(rx_mbufs[i]);
                }
            }
        }
        
        /* Calculate processing time */
        uint64_t end_time = get_timestamp_us();
        uint64_t processing_time = end_time - start_time;
        
        /* Update metrics for each port */
        for (uint16_t port = 0; port < port_count; port++) {
            if (rx_packets_by_port[port] > 0) {
                dpdk_metrics_update(&g_metrics, port, 
                                  rx_packets_by_port[port], rx_bytes_by_port[port],
                                  rx_packets_by_port[port], processing_time);
            }
        }
    }
    
    printf("\nCleaning up...\n");
    
    /* Set quit flag for CUDA kernel */
    cudaMemset(d_quit_flag, 1, sizeof(uint32_t));
    
    /* Wait for CUDA kernel to complete */
    cudaStreamSynchronize(cuda_stream);
    
    /* Print metrics */
    dpdk_metrics_print(&g_metrics);
    
    /* Cleanup resources */
    dpdk_metrics_cleanup(&g_metrics);
    cudaFree(comm_list);
    cudaFree(d_quit_flag);
    cudaStreamDestroy(cuda_stream);
    
    /* Clean up the driver */
    dpdk_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 
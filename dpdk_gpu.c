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
#include <rte_gpudev.h>

#include "dpdk_driver.h"
#include "dpdk_gpu.h"

//////////////////////////////////////////////////////////////////////////
///// GPU processing definitions
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
    
    struct rte_gpu_comm_flag quit_flag;
    struct rte_gpu_comm_list *comm_list = NULL;
    int nb_rx = 0;
    int queue_id = 0;
    struct rte_mbuf *rx_mbufs[MAX_RX_MBUFS];
    cudaStream_t cuda_stream;
    int16_t gpu_dev_id = 0;
    
    /* Initialize CUDA */
    ret = init_cuda_device(gpu_dev_id);
    if (ret != 0) {
        printf("Failed to set CUDA device\n");
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    ret = create_cuda_stream(&cuda_stream);
    if (ret != 0) {
        printf("Failed to create CUDA stream\n");
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Create communication flag using DPDK GPU API */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    ret = rte_gpu_comm_create_flag(gpu_dev_id, &quit_flag, RTE_GPU_COMM_FLAG_CPU);
    if (ret != 0) {
        printf("Failed to create GPU communication flag: %d\n", ret);
        destroy_cuda_stream(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Set initial flag value to 0 */
    ret = rte_gpu_comm_set_flag(&quit_flag, 0);
    if (ret != 0) {
        printf("Failed to set GPU flag: %d\n", ret);
        rte_gpu_comm_destroy_flag(&quit_flag);
        destroy_cuda_stream(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Create communication list using DPDK GPU API */
    comm_list = rte_gpu_comm_create_list(gpu_dev_id, COMM_LIST_ENTRIES);
    if (comm_list == NULL) {
        printf("Failed to create GPU communication list: %d\n", rte_errno);
        rte_gpu_comm_destroy_flag(&quit_flag);
        destroy_cuda_stream(cuda_stream);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
#pragma GCC diagnostic pop

    /* Start CUDA kernel for packet processing */
    launch_packet_processing(quit_flag.ptr, comm_list, COMM_LIST_ENTRIES, cuda_stream);
    
    printf("GPU packet processing started. Press Ctrl+C to exit...\n");
    
    /* Arrays for metrics calculation */
    uint32_t rx_packets_by_port[MAX_PORTS] = {0};
    uint64_t rx_bytes_by_port[MAX_PORTS] = {0};
    int current_list = 0;
    
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
                
                /* Check if current list is free */
                enum rte_gpu_comm_list_status status;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                rte_gpu_comm_get_status(&comm_list[current_list], &status);
                
                if (status == RTE_GPU_COMM_LIST_FREE) {
                    /* Populate the list with packet data and mark as ready for GPU */
                    rte_gpu_comm_populate_list_pkts(&comm_list[current_list], rx_mbufs, nb_rx);
                    rte_gpu_comm_set_status(&comm_list[current_list], RTE_GPU_COMM_LIST_READY);
                    
                    /* Ensure memory coherency */
                    rte_gpu_wmb(gpu_dev_id);
                    
                    /* Move to next list */
                    current_list = (current_list + 1) % COMM_LIST_ENTRIES;
                } else {
                    /* Free mbufs if we can't process them (we don't want any CPU fallback) */
                    for (int i = 0; i < nb_rx; i++) {
                        rte_pktmbuf_free(rx_mbufs[i]);
                    }
                    printf("Warning: Dropped packets due to no available GPU processing slots\n");
                }
            }
            
            /* Check for completed processing in GPU */
            for (int i = 0; i < COMM_LIST_ENTRIES; i++) {
                if (i != current_list) {
                    enum rte_gpu_comm_list_status status;
                    rte_gpu_comm_get_status(&comm_list[i], &status);
                    
                    if (status == RTE_GPU_COMM_LIST_DONE) {
                        /* Cleanup list and free mbufs */
                        rte_gpu_comm_cleanup_list(&comm_list[i]);
                    }
                }
            }
#pragma GCC diagnostic pop
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rte_gpu_comm_set_flag(&quit_flag, 1);
#pragma GCC diagnostic pop
    
    /* Wait for CUDA kernel to complete */
    sync_cuda_stream(cuda_stream);
    
    /* Print metrics */
    dpdk_metrics_print(&g_metrics);
    
    /* Cleanup resources */
    dpdk_metrics_cleanup(&g_metrics);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rte_gpu_comm_destroy_list(comm_list, COMM_LIST_ENTRIES);
    rte_gpu_comm_destroy_flag(&quit_flag);
#pragma GCC diagnostic pop
    destroy_cuda_stream(cuda_stream);
    
    /* Clean up the driver */
    dpdk_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 
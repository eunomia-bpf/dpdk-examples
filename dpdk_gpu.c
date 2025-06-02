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

/* GPU processing data structures */
typedef struct {
    /* Host memory */
    void *h_pkt_addrs[MAX_RX_MBUFS];
    uint32_t h_pkt_lengths[MAX_RX_MBUFS];
    uint32_t h_pkt_count;
    uint32_t h_status_flag;
    
    /* Device memory */
    uint32_t *d_quit_flag;
    void **d_pkt_addrs;
    uint32_t *d_pkt_lengths;
    uint32_t *d_pkt_count;
    uint32_t *d_status_flag;
} gpu_data_t;

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

/* Initialize GPU device */
static int init_gpu_data(gpu_data_t *gpu_data)
{
    int ret;
    
    /* Initialize CUDA with device 0 */
    ret = init_cuda_device(0);
    if (ret != 0) {
        printf("Failed to initialize CUDA device\n");
        return -1;
    }
    
    printf("CUDA device initialized successfully\n");
    
    /* Initialize host memory */
    memset(gpu_data->h_pkt_addrs, 0, sizeof(gpu_data->h_pkt_addrs));
    memset(gpu_data->h_pkt_lengths, 0, sizeof(gpu_data->h_pkt_lengths));
    gpu_data->h_pkt_count = 0;
    gpu_data->h_status_flag = 0;
    
    /* Allocate GPU memory for quit flag */
    ret = alloc_gpu_memory((void **)&gpu_data->d_quit_flag, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to allocate GPU memory for quit flag\n");
        return -1;
    }
    
    /* Set quit flag to 0 (running) */
    ret = set_gpu_memory(gpu_data->d_quit_flag, 0, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to initialize quit flag\n");
        free_gpu_memory(gpu_data->d_quit_flag);
        return -1;
    }
    
    /* Allocate GPU memory for packet addresses */
    ret = alloc_gpu_memory((void **)&gpu_data->d_pkt_addrs, MAX_RX_MBUFS * sizeof(void *));
    if (ret != 0) {
        printf("Failed to allocate GPU memory for packet addresses\n");
        free_gpu_memory(gpu_data->d_quit_flag);
        return -1;
    }
    
    /* Allocate GPU memory for packet lengths */
    ret = alloc_gpu_memory((void **)&gpu_data->d_pkt_lengths, MAX_RX_MBUFS * sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to allocate GPU memory for packet lengths\n");
        free_gpu_memory(gpu_data->d_pkt_addrs);
        free_gpu_memory(gpu_data->d_quit_flag);
        return -1;
    }
    
    /* Allocate GPU memory for packet count */
    ret = alloc_gpu_memory((void **)&gpu_data->d_pkt_count, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to allocate GPU memory for packet count\n");
        free_gpu_memory(gpu_data->d_pkt_lengths);
        free_gpu_memory(gpu_data->d_pkt_addrs);
        free_gpu_memory(gpu_data->d_quit_flag);
        return -1;
    }
    
    /* Allocate GPU memory for status flag */
    ret = alloc_gpu_memory((void **)&gpu_data->d_status_flag, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to allocate GPU memory for status flag\n");
        free_gpu_memory(gpu_data->d_pkt_count);
        free_gpu_memory(gpu_data->d_pkt_lengths);
        free_gpu_memory(gpu_data->d_pkt_addrs);
        free_gpu_memory(gpu_data->d_quit_flag);
        return -1;
    }
    
    /* Initialize device memory */
    ret = set_gpu_memory(gpu_data->d_pkt_addrs, 0, MAX_RX_MBUFS * sizeof(void *));
    if (ret != 0) {
        printf("Failed to initialize packet addresses\n");
        goto cleanup;
    }
    
    ret = set_gpu_memory(gpu_data->d_pkt_lengths, 0, MAX_RX_MBUFS * sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to initialize packet lengths\n");
        goto cleanup;
    }
    
    ret = set_gpu_memory(gpu_data->d_pkt_count, 0, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to initialize packet count\n");
        goto cleanup;
    }
    
    ret = set_gpu_memory(gpu_data->d_status_flag, 0, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to initialize status flag\n");
        goto cleanup;
    }
    
    printf("Successfully initialized all GPU memory\n");
    return 0;
    
cleanup:
    free_gpu_memory(gpu_data->d_status_flag);
    free_gpu_memory(gpu_data->d_pkt_count);
    free_gpu_memory(gpu_data->d_pkt_lengths);
    free_gpu_memory(gpu_data->d_pkt_addrs);
    free_gpu_memory(gpu_data->d_quit_flag);
    return -1;
}

/* Clean up GPU data */
static void cleanup_gpu_data(gpu_data_t *gpu_data)
{
    if (gpu_data->d_status_flag)
        free_gpu_memory(gpu_data->d_status_flag);
    
    if (gpu_data->d_pkt_count)
        free_gpu_memory(gpu_data->d_pkt_count);
    
    if (gpu_data->d_pkt_lengths)
        free_gpu_memory(gpu_data->d_pkt_lengths);
    
    if (gpu_data->d_pkt_addrs)
        free_gpu_memory(gpu_data->d_pkt_addrs);
    
    if (gpu_data->d_quit_flag)
        free_gpu_memory(gpu_data->d_quit_flag);
    
    printf("GPU memory resources cleaned up\n");
}

/* Process packets on GPU - simplified direct approach */
static int process_packets_on_gpu(gpu_data_t *gpu_data, struct rte_mbuf **mbufs, int nb_pkts, cudaStream_t stream)
{
    int ret;
    
    /* Validate input */
    if (nb_pkts <= 0 || nb_pkts > MAX_RX_MBUFS) {
        printf("Invalid packet count: %d\n", nb_pkts);
        return -1;
    }
    
    /* Prepare packet data in host memory first */
    for (int i = 0; i < nb_pkts; i++) {
        gpu_data->h_pkt_addrs[i] = rte_pktmbuf_mtod(mbufs[i], void *);
        gpu_data->h_pkt_lengths[i] = rte_pktmbuf_pkt_len(mbufs[i]);
        
        /* Simple validation */
        if (gpu_data->h_pkt_addrs[i] == NULL) {
            printf("Warning: NULL packet address at index %d\n", i);
        }
    }
    
    /* Register packet data memory with CUDA to make it accessible from GPU */
    for (int i = 0; i < nb_pkts; i++) {
        ret = register_cpu_to_gpu_memory(gpu_data->h_pkt_addrs[i], gpu_data->h_pkt_lengths[i]);
        if (ret != 0) {
            printf("Failed to register packet memory at index %d\n", i);
            return -1;
        }
    }
    
    /* Set packet count */
    gpu_data->h_pkt_count = nb_pkts;
    ret = copy_to_gpu(gpu_data->d_pkt_count, &gpu_data->h_pkt_count, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to copy packet count to GPU: %d\n", ret);
        return -1;
    }
    
    /* Copy packet addresses to GPU */
    ret = copy_to_gpu(gpu_data->d_pkt_addrs, gpu_data->h_pkt_addrs, nb_pkts * sizeof(void *));
    if (ret != 0) {
        printf("Failed to copy packet addresses to GPU: %d\n", ret);
        return -1;
    }
    
    /* Copy packet lengths to GPU */
    ret = copy_to_gpu(gpu_data->d_pkt_lengths, gpu_data->h_pkt_lengths, nb_pkts * sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to copy packet lengths to GPU: %d\n", ret);
        return -1;
    }
    
    /* Set status flag to 1 (ready for processing) */
    gpu_data->h_status_flag = 1;
    ret = copy_to_gpu(gpu_data->d_status_flag, &gpu_data->h_status_flag, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to set status flag: %d\n", ret);
        return -1;
    }
    
    /* Launch the packet processing kernel */
    launch_simple_packet_processing(
        gpu_data->d_quit_flag,
        gpu_data->d_pkt_addrs,
        gpu_data->d_pkt_lengths,
        gpu_data->d_pkt_count,
        gpu_data->d_status_flag,
        stream
    );
    
    /* Wait for kernel to complete processing */
    ret = synchronize_gpu_stream(stream);
    if (ret != 0) {
        printf("Failed to synchronize GPU stream: %d\n", ret);
        return -1;
    }
    
    /* Check status flag to make sure processing is complete */
    ret = copy_from_gpu(&gpu_data->h_status_flag, gpu_data->d_status_flag, sizeof(uint32_t));
    if (ret != 0) {
        printf("Failed to get status flag from GPU: %d\n", ret);
        return -1;
    }
    
    if (gpu_data->h_status_flag != 0) {
        printf("GPU processing not completed, status=%u\n", gpu_data->h_status_flag);
        return -1;
    }
    
    /* Unregister packet memory */
    for (int i = 0; i < nb_pkts; i++) {
        unregister_cpu_memory(gpu_data->h_pkt_addrs[i]);
    }
    
    return 0;
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
    
    int nb_rx = 0;
    int queue_id = 0;
    struct rte_mbuf *rx_mbufs[MAX_RX_MBUFS];
    cudaStream_t cuda_stream;
    gpu_data_t gpu_data = {0};
    
    /* Initialize GPU data */
    ret = init_gpu_data(&gpu_data);
    if (ret != 0) {
        printf("Failed to initialize GPU data\n");
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Create CUDA stream */
    ret = create_cuda_stream(&cuda_stream);
    if (ret != 0) {
        printf("Failed to create CUDA stream\n");
        cleanup_gpu_data(&gpu_data);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    printf("GPU packet processing started. Press Ctrl+C to exit...\n");
    
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
                
                /* Process packets on GPU */
                ret = process_packets_on_gpu(&gpu_data, rx_mbufs, nb_rx, cuda_stream);
                if (ret != 0) {
                    printf("Warning: GPU processing failed, packets will be dropped\n");
                }
                
                /* Free the mbufs */
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
    
    /* Set quit flag for GPU processing */
    uint32_t quit = 1;
    copy_to_gpu(gpu_data.d_quit_flag, &quit, sizeof(uint32_t));
    
    /* Wait for GPU to complete any pending work */
    synchronize_gpu_stream(cuda_stream);
    
    /* Print metrics */
    dpdk_metrics_print(&g_metrics);
    
    /* Cleanup resources */
    dpdk_metrics_cleanup(&g_metrics);
    destroy_cuda_stream(cuda_stream);
    cleanup_gpu_data(&gpu_data);
    
    /* Clean up the driver */
    dpdk_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 
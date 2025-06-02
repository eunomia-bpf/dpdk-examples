#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

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

/* CUDA kernel for packet processing */
__global__ void cuda_kernel_packet_processing(uint32_t *quit_flag_ptr, 
                                            struct rte_gpu_comm_list *comm_list, 
                                            int comm_list_entries)
{
    int comm_list_index = 0;
    
    /* Do some pre-processing operations. */
    
    /* GPU kernel keeps checking this flag to know if it has to quit or wait for more packets. */
    while (*quit_flag_ptr == 0) {
        if (comm_list[comm_list_index].status_d[0] != RTE_GPU_COMM_LIST_READY)
            continue;
            
        if (threadIdx.x < comm_list[comm_list_index].num_pkts)
        {
            /* Each CUDA thread processes a different packet. */
            /* packet_processing(comm_list[comm_list_index].addr, comm_list[comm_list_index].size, ..); */
            
            /* Simple packet processing - just incrementing a counter in the packet */
            uint8_t *pkt_data = (uint8_t *)comm_list[comm_list_index].addr[threadIdx.x];
            if (pkt_data != NULL) {
                /* Simple operation - increment first byte */
                pkt_data[0]++;
            }
        }
        __threadfence();
        __syncthreads();
        
        /* Wait for new packets on the next communication list entry. */
        comm_list_index = (comm_list_index+1) % comm_list_entries;
    }
    
    /* Do some post-processing operations. */
}

int main(int argc, char *argv[])
{
    /* Install signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("DPDK GPU Packet Processing Application\n");
    printf("======================================\n");
    
    /* Initialize the DPDK driver */
    dpdk_config_t config = DPDK_DEFAULT_CONFIG;
    config.burst_size = MAX_RX_MBUFS;  /* Maximum burst size */
    
    int ret = dpdk_init(argc, argv, &config);
    if (ret != 0) {
        printf("Failed to initialize DPDK: %d\n", ret);
        exit(EXIT_FAILURE);
    }
    
    /* Initialize metrics */
    uint16_t port_count = dpdk_get_port_count();
    if (port_count == 0) {
        printf("No ports found! Make sure to use --vdev option.\n");
        printf("Examples:\n");
        printf("  %s --vdev=net_null0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_tap0,iface=test0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_ring0 -l 0\n", argv[0]);
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    ret = dpdk_metrics_init(&g_metrics, port_count);
    if (ret != 0) {
        printf("Failed to initialize metrics: %d\n", ret);
        dpdk_cleanup();
        exit(EXIT_FAILURE);
    }
    
    struct rte_gpu_flag quit_flag;
    struct rte_gpu_comm_list *comm_list;
    int nb_rx = 0;
    int comm_list_entry = 0;
    struct rte_mbuf *rx_mbufs[MAX_RX_MBUFS];
    cudaStream_t cstream;
    struct rte_mempool *mpool_payload, *mpool_header;
    struct rte_pktmbuf_extmem ext_mem;
    int16_t dev_id;
    int16_t port_id = 0;
    uint16_t queue_id = 0;
    
    /* Initialize CUDA objects (cstream, context, etc..). */
    cudaError_t cuda_status = cudaStreamCreate(&cstream);
    if (cuda_status != cudaSuccess) {
        printf("Failed to create CUDA stream: %s\n", cudaGetErrorString(cuda_status));
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* Let's assume the application wants to use the default context of the GPU device 0. */
    dev_id = 0;
    
    /* Create an external memory mempool using memory allocated on the GPU. */
    ext_mem.elt_size = MBUFS_HEADROOM_SIZE;
    ext_mem.buf_len = RTE_ALIGN_CEIL(MBUFS_NUM * ext_mem.elt_size, GPU_PAGE_SIZE);
    ext_mem.buf_iova = RTE_BAD_IOVA;
    ext_mem.buf_ptr = dpdk_gpu_mem_alloc(dev_id, ext_mem.buf_len, 0);
    if (ext_mem.buf_ptr == NULL) {
        printf("Failed to allocate GPU memory\n");
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    ret = dpdk_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova, GPU_PAGE_SIZE);
    if (ret != 0) {
        printf("Failed to register external memory: %d\n", ret);
        dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    ret = dpdk_dev_dma_map(port_id, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
    if (ret != 0) {
        printf("Failed to map DMA memory: %d\n", ret);
        dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    mpool_payload = rte_pktmbuf_pool_create_extbuf("gpu_mempool", MBUFS_NUM,
                                                   0, 0, ext_mem.elt_size,
                                                   rte_socket_id(), &ext_mem, 1);
    if (mpool_payload == NULL) {
        printf("Failed to create mempool: %s\n", rte_strerror(rte_errno));
        dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /*
     * Create CPU - device communication flag.
     * With this flag, the CPU can tell to the CUDA kernel to exit from the main loop.
     */
    ret = dpdk_gpu_comm_create_flag(dev_id, &quit_flag, RTE_GPU_COMM_FLAG_CPU);
    if (ret != 0) {
        printf("Failed to create GPU communication flag: %d\n", ret);
        dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    dpdk_gpu_comm_set_flag(&quit_flag, 0);
    
    /*
     * Create CPU - device communication list.
     * Each entry of this list will be populated by the CPU
     * with a new set of received mbufs that the CUDA kernel has to process.
     */
    comm_list = dpdk_gpu_comm_create_list(dev_id, COMM_LIST_ENTRIES);
    if (comm_list == NULL) {
        printf("Failed to create GPU communication list\n");
        dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
        dpdk_metrics_cleanup(&g_metrics);
        dpdk_cleanup();
        return EXIT_FAILURE;
    }
    
    /* A very simple CUDA kernel with just 1 CUDA block and RTE_GPU_COMM_LIST_PKTS_MAX CUDA threads. */
    cuda_kernel_packet_processing<<<1, RTE_GPU_COMM_LIST_PKTS_MAX, 0, cstream>>>(
        quit_flag.ptr, comm_list, COMM_LIST_ENTRIES);
    
    printf("Packet processing running on GPU. Press Ctrl+C to exit...\n");
    
    /* Define arrays for metrics calculation */
    uint32_t rx_packets_by_port[MAX_PORTS] = {0};
    uint64_t rx_bytes_by_port[MAX_PORTS] = {0};
    
    /* Main processing loop */
    while (!force_quit) {
        /* Track processing time */
        uint64_t start_time = get_timestamp_us();
        
        /* Process each port */
        for (uint16_t port = 0; port < port_count; port++) {
            /* Reset counters for this iteration */
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
                
                /* Send packets to GPU for processing */
                dpdk_gpu_comm_populate_list_pkts(&comm_list[comm_list_entry], (void **)rx_mbufs, nb_rx);
                
                /* Move to next entry in comm list */
                comm_list_entry = (comm_list_entry + 1) % COMM_LIST_ENTRIES;
                
                /* Wait for GPU to process previous batches if needed */
                if (comm_list_entry % 2 == 0) {
                    int prev_entry = (comm_list_entry - 2 + COMM_LIST_ENTRIES) % COMM_LIST_ENTRIES;
                    while (dpdk_gpu_comm_cleanup_list(&comm_list[prev_entry])) {
                        /* Wait for cleanup */
                    }
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
    
    /* Cleanup any remaining comm list entries */
    for (int i = 0; i < COMM_LIST_ENTRIES; i++) {
        while (dpdk_gpu_comm_cleanup_list(&comm_list[i])) {
            /* Wait for cleanup */
        }
    }
    
    /* CPU notifies the CUDA kernel that it has to terminate. */
    dpdk_gpu_comm_set_flag(&quit_flag, 1);
    
    /* Wait for CUDA kernel to complete */
    cudaStreamSynchronize(cstream);
    
    /* Print metrics */
    dpdk_metrics_print(&g_metrics);
    
    /* Cleanup resources */
    dpdk_metrics_cleanup(&g_metrics);
    dpdk_gpu_mem_free(dev_id, ext_mem.buf_ptr);
    cudaStreamDestroy(cstream);
    
    /* Clean up the driver */
    dpdk_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 
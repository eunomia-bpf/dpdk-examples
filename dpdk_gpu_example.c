#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <stdbool.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_gpu.h>
#include <rte_gpu_comm.h>

#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////////////////
///// gpudev library + CUDA functions
//////////////////////////////////////////////////////////////////////////
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)

#define MAX_RX_MBUFS 32
#define MBUFS_NUM 1024
#define MBUFS_HEADROOM_SIZE 256
#define COMM_LIST_ENTRIES 8

/* Signal handler */
static volatile bool force_quit = false;

static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
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
    
    /* Initialize the EAL */
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
    
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
        return EXIT_FAILURE;
    }
    
    /* Let's assume the application wants to use the default context of the GPU device 0. */
    dev_id = 0;
    
    /* Create an external memory mempool using memory allocated on the GPU. */
    ext_mem.elt_size = MBUFS_HEADROOM_SIZE;
    ext_mem.buf_len = RTE_ALIGN_CEIL(MBUFS_NUM * ext_mem.elt_size, GPU_PAGE_SIZE);
    ext_mem.buf_iova = RTE_BAD_IOVA;
    ext_mem.buf_ptr = rte_gpu_mem_alloc(dev_id, ext_mem.buf_len, 0);
    if (ext_mem.buf_ptr == NULL) {
        printf("Failed to allocate GPU memory\n");
        return EXIT_FAILURE;
    }
    
    ret = rte_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova, GPU_PAGE_SIZE);
    if (ret != 0) {
        printf("Failed to register external memory: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    ret = rte_dev_dma_map(rte_eth_devices[port_id].device,
            ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
    if (ret != 0) {
        printf("Failed to map DMA memory: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    mpool_payload = rte_pktmbuf_pool_create_extbuf("gpu_mempool", MBUFS_NUM,
                                                   0, 0, ext_mem.elt_size,
                                                   rte_socket_id(), &ext_mem, 1);
    if (mpool_payload == NULL) {
        printf("Failed to create mempool: %s\n", rte_strerror(rte_errno));
        return EXIT_FAILURE;
    }
    
    /*
     * Create CPU - device communication flag.
     * With this flag, the CPU can tell to the CUDA kernel to exit from the main loop.
     */
    ret = rte_gpu_comm_create_flag(dev_id, &quit_flag, RTE_GPU_COMM_FLAG_CPU);
    if (ret != 0) {
        printf("Failed to create GPU communication flag: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    rte_gpu_comm_set_flag(&quit_flag, 0);
    
    /*
     * Create CPU - device communication list.
     * Each entry of this list will be populated by the CPU
     * with a new set of received mbufs that the CUDA kernel has to process.
     */
    comm_list = rte_gpu_comm_create_list(dev_id, COMM_LIST_ENTRIES);
    if (comm_list == NULL) {
        printf("Failed to create GPU communication list\n");
        return EXIT_FAILURE;
    }
    
    /* Configure and start the port */
    ret = rte_eth_dev_configure(port_id, 1, 1, NULL);
    if (ret < 0) {
        printf("Failed to configure Ethernet device: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    ret = rte_eth_rx_queue_setup(port_id, queue_id, 128, rte_eth_dev_socket_id(port_id), NULL, mpool_payload);
    if (ret < 0) {
        printf("Failed to setup RX queue: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    ret = rte_eth_tx_queue_setup(port_id, queue_id, 128, rte_eth_dev_socket_id(port_id), NULL);
    if (ret < 0) {
        printf("Failed to setup TX queue: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        printf("Failed to start Ethernet device: %d\n", ret);
        return EXIT_FAILURE;
    }
    
    /* A very simple CUDA kernel with just 1 CUDA block and RTE_GPU_COMM_LIST_PKTS_MAX CUDA threads. */
    cuda_kernel_packet_processing<<<1, RTE_GPU_COMM_LIST_PKTS_MAX, 0, cstream>>>(
        quit_flag.ptr, comm_list, COMM_LIST_ENTRIES);
    
    printf("Packet processing running on GPU. Press Ctrl+C to exit...\n");
    
    /* Main processing loop */
    while (!force_quit) {
        /* Receive packets */
        nb_rx = rte_eth_rx_burst(port_id, queue_id, rx_mbufs, MAX_RX_MBUFS);
        if (nb_rx > 0) {
            /* Send packets to GPU for processing */
            rte_gpu_comm_populate_list_pkts(&comm_list[comm_list_entry], rx_mbufs, nb_rx);
            
            /* Move to next entry in comm list */
            comm_list_entry = (comm_list_entry + 1) % COMM_LIST_ENTRIES;
            
            /* Wait for GPU to process previous batches if needed */
            if (comm_list_entry % 2 == 0) {
                int prev_entry = (comm_list_entry - 2 + COMM_LIST_ENTRIES) % COMM_LIST_ENTRIES;
                while (rte_gpu_comm_cleanup_list(&comm_list[prev_entry])) {
                    /* Wait for cleanup */
                }
            }
        }
    }
    
    printf("\nCleaning up...\n");
    
    /* Cleanup any remaining comm list entries */
    for (int i = 0; i < COMM_LIST_ENTRIES; i++) {
        while (rte_gpu_comm_cleanup_list(&comm_list[i])) {
            /* Wait for cleanup */
        }
    }
    
    /* CPU notifies the CUDA kernel that it has to terminate. */
    rte_gpu_comm_set_flag(&quit_flag, 1);
    
    /* Wait for CUDA kernel to complete */
    cudaStreamSynchronize(cstream);
    
    /* Stop the port and release resources */
    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    
    /* gpudev objects cleanup/destruction */
    rte_gpu_mem_free(dev_id, ext_mem.buf_ptr);
    cudaStreamDestroy(cstream);
    
    /* Clean up the EAL */
    rte_eal_cleanup();
    
    printf("Goodbye!\n");
    return 0;
} 
#ifndef DPDK_DRIVER_H
#define DPDK_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>  /* For size_t definition */

/* Check if GPU support is available */
#if defined(RTE_HAS_GPU_SUPPORT)
#define DPDK_GPU_SUPPORT 1
#else
/* Try to include the header and see if it works */
#if __has_include(<rte_gpu.h>)
#define DPDK_GPU_SUPPORT 1
#else
#define DPDK_GPU_SUPPORT 0
#endif
#endif

/* Driver configuration */
typedef struct {
    bool promiscuous_mode;     /* Enable promiscuous mode */
    uint16_t rx_ring_size;     /* RX ring buffer size */
    uint16_t tx_ring_size;     /* TX ring buffer size */
    uint16_t burst_size;       /* Maximum burst size */
} dpdk_config_t;

/* Default configuration values */
extern const dpdk_config_t DPDK_DEFAULT_CONFIG;

/* Packet data structure */
typedef struct {
    uint16_t port;             /* Port the packet was received on */
    void *data;                /* Packet data */
    uint32_t length;           /* Packet length in bytes */
} dpdk_packet_t;

/**
 * Initialize the DPDK environment and ports
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @param config Driver configuration (NULL for default)
 * @return 0 on success, negative on error
 */
int dpdk_init(int argc, char *argv[], const dpdk_config_t *config);

/**
 * Get the number of available DPDK ports
 * 
 * @return Number of ports
 */
uint16_t dpdk_get_port_count(void);

/**
 * Poll for packets on all ports
 * 
 * @param pkts Array to store received packets
 * @param max_pkts Maximum number of packets to receive
 * @param bytes_received Pointer to store total bytes received (can be NULL)
 * @return Number of packets received, negative on error
 */
int dpdk_poll(dpdk_packet_t *pkts, uint16_t max_pkts, uint64_t *bytes_received);

/**
 * Free a packet
 * 
 * @param pkt Packet to free
 */
void dpdk_free_packet(dpdk_packet_t *pkt);

/**
 * Free multiple packets
 * 
 * @param pkts Array of packets to free
 * @param count Number of packets to free
 */
void dpdk_free_packets(dpdk_packet_t *pkts, int count);

/**
 * Cleanup DPDK environment
 */
void dpdk_cleanup(void);

/* Simple metrics API */
typedef struct {
    uint64_t rx_packets;
    uint64_t rx_bytes;
    uint64_t processed_packets;
    uint64_t processing_time_us;
} dpdk_port_metrics_t;

typedef struct {
    uint64_t start_time_sec;
    uint64_t total_rx_packets;
    uint64_t total_rx_bytes;
    uint64_t total_processed_packets;
    uint64_t total_processing_time_us;
    dpdk_port_metrics_t *ports;
} dpdk_metrics_t;

/**
 * Initialize metrics collection
 *
 * @param metrics Pointer to metrics structure
 * @param num_ports Number of ports to track
 * @return 0 on success, negative on error
 */
int dpdk_metrics_init(dpdk_metrics_t *metrics, uint16_t num_ports);

/**
 * Update port metrics
 *
 * @param metrics Pointer to metrics structure
 * @param port Port ID
 * @param rx_packets Number of received packets
 * @param rx_bytes Number of received bytes
 * @param processed_packets Number of processed packets
 * @param processing_time_us Processing time in microseconds
 */
void dpdk_metrics_update(dpdk_metrics_t *metrics, uint16_t port, 
                        uint32_t rx_packets, uint64_t rx_bytes,
                        uint32_t processed_packets, uint64_t processing_time_us);

/**
 * Print metrics summary
 *
 * @param metrics Pointer to metrics structure
 */
void dpdk_metrics_print(dpdk_metrics_t *metrics);

/**
 * Cleanup metrics
 *
 * @param metrics Pointer to metrics structure
 */
void dpdk_metrics_cleanup(dpdk_metrics_t *metrics);

/* GPU-related API - only available if DPDK_GPU_SUPPORT is enabled */
#if DPDK_GPU_SUPPORT

/**
 * Allocate memory on the GPU device
 *
 * @param dev_id GPU device ID
 * @param size Size of the memory to allocate
 * @param flags Allocation flags
 * @return Pointer to allocated memory or NULL on failure
 */
void* dpdk_gpu_mem_alloc(int16_t dev_id, size_t size, unsigned int flags);

/**
 * Free memory allocated on the GPU device
 *
 * @param dev_id GPU device ID
 * @param ptr Pointer to memory to free
 * @return 0 on success, negative on error
 */
int dpdk_gpu_mem_free(int16_t dev_id, void *ptr);

/**
 * Register a communication flag for CPU-GPU synchronization
 *
 * @param dev_id GPU device ID
 * @param flag Pointer to store the created flag
 * @param flag_type Flag type (CPU or GPU managed)
 * @return 0 on success, negative on error
 */
int dpdk_gpu_comm_create_flag(int16_t dev_id, void *flag, int flag_type);

/**
 * Set value of a communication flag
 *
 * @param flag The flag to modify
 * @param value Value to set
 * @return 0 on success, negative on error
 */
int dpdk_gpu_comm_set_flag(void *flag, uint32_t value);

/**
 * Create a communication list for packet data exchange between CPU and GPU
 *
 * @param dev_id GPU device ID
 * @param num_entries Number of entries in the communication list
 * @return Pointer to created communication list or NULL on failure
 */
void* dpdk_gpu_comm_create_list(int16_t dev_id, int num_entries);

/**
 * Populate a communication list entry with packet data
 *
 * @param comm_list Communication list entry to populate
 * @param pkts Array of packet pointers
 * @param nb_pkts Number of packets
 * @return 0 on success, negative on error
 */
int dpdk_gpu_comm_populate_list_pkts(void *comm_list, void **pkts, int nb_pkts);

/**
 * Cleanup a communication list entry after GPU processing
 *
 * @param comm_list Communication list entry to clean
 * @return 0 when cleanup is complete, 1 if more cleanup needed
 */
int dpdk_gpu_comm_cleanup_list(void *comm_list);

/**
 * Register an external memory region for DPDK
 *
 * @param addr Memory address
 * @param len Memory region length
 * @param context Context (optional)
 * @param iova IOVA address
 * @param page_size Page size for the memory
 * @return 0 on success, negative on error
 */
int dpdk_extmem_register(void *addr, size_t len, void *context, uint64_t iova, size_t page_size);

/**
 * Map a memory region for DMA operations
 *
 * @param port_id Port ID
 * @param addr Memory address
 * @param iova IOVA address
 * @param len Memory region length
 * @return 0 on success, negative on error
 */
int dpdk_dev_dma_map(uint16_t port_id, void *addr, uint64_t iova, size_t len);

#endif /* DPDK_GPU_SUPPORT */

#endif /* DPDK_DRIVER_H */ 
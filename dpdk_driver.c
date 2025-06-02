#include "dpdk_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>  /* For PRIx8 format specifier */
#include <time.h>      /* For clock_gettime() */

/* DPDK headers */
#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

/* Default configuration values */
#define DEFAULT_RX_RING_SIZE 1024
#define DEFAULT_TX_RING_SIZE 1024
#define DEFAULT_BURST_SIZE 32
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250

/* Default configuration */
const dpdk_config_t DPDK_DEFAULT_CONFIG = {
    .promiscuous_mode = true,
    .rx_ring_size = DEFAULT_RX_RING_SIZE,
    .tx_ring_size = DEFAULT_TX_RING_SIZE,
    .burst_size = DEFAULT_BURST_SIZE
};

/* Global driver state */
static bool g_initialized = false;
static struct rte_mempool *g_mbuf_pool = NULL;
static dpdk_config_t g_config;
static uint16_t g_nb_ports = 0;

/* Port configuration */
static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        /* Remove deprecated max_rx_pkt_len field */
    },
};

/**
 * Initialize a single port
 */
static int port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    uint16_t nb_rxd = g_config.rx_ring_size;
    uint16_t nb_txd = g_config.tx_ring_size;
    int retval;
    uint16_t q;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf txconf;

    if (!rte_eth_dev_is_valid_port(port))
        return -1;

    retval = rte_eth_dev_info_get(port, &dev_info);
    if (retval != 0) {
        printf("Error during getting device (port %u) info: %s\n",
                port, strerror(-retval));
        return retval;
    }

    printf("Device info: driver=%s\n", dev_info.driver_name);

    /* Configure the Ethernet device */
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        printf("Failed to configure port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (retval != 0) {
        printf("Failed to adjust descriptors for port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    /* Allocate and set up 1 RX queue per Ethernet port */
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
                rte_eth_dev_socket_id(port), NULL, mbuf_pool);
        if (retval < 0) {
            printf("Failed to setup RX queue %u for port %u: %s\n", 
                   q, port, strerror(-retval));
            return retval;
        }
    }

    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    /* Allocate and set up 1 TX queue per Ethernet port */
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd,
                rte_eth_dev_socket_id(port), &txconf);
        if (retval < 0) {
            printf("Failed to setup TX queue %u for port %u: %s\n", 
                   q, port, strerror(-retval));
            return retval;
        }
    }

    /* Start the Ethernet port */
    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        printf("Failed to start port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    /* Display the port MAC address */
    struct rte_ether_addr addr;
    retval = rte_eth_macaddr_get(port, &addr);
    if (retval != 0) {
        printf("Failed to get MAC address for port %u: %s\n", port, strerror(-retval));
        return retval;
    }

    printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
           " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
           port,
           addr.addr_bytes[0], addr.addr_bytes[1],
           addr.addr_bytes[2], addr.addr_bytes[3],
           addr.addr_bytes[4], addr.addr_bytes[5]);

    /* Enable RX in promiscuous mode if configured */
    if (g_config.promiscuous_mode) {
        retval = rte_eth_promiscuous_enable(port);
        if (retval != 0)
            printf("Warning: failed to enable promiscuous mode for port %u: %s\n",
                   port, strerror(-retval));
    }

    return 0;
}

int dpdk_init(int argc, char *argv[], const dpdk_config_t *config)
{
    if (g_initialized) {
        printf("DPDK already initialized\n");
        return -EALREADY;
    }

    /* Use default config if none provided */
    if (config) {
        g_config = *config;
    } else {
        g_config = DPDK_DEFAULT_CONFIG;
    }

    /* Initialize the Environment Abstraction Layer (EAL) */
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        printf("Error with EAL initialization\n");
        return -1;
    }

    /* Check that there are ports available */
    g_nb_ports = rte_eth_dev_count_avail();
    printf("Found %u ports\n", g_nb_ports);

    if (g_nb_ports == 0) {
        printf("No ports found! Make sure to use --vdev option.\n");
        printf("Examples:\n");
        printf("  %s --vdev=net_null0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_tap0,iface=test0 -l 0\n", argv[0]);
        printf("  %s --vdev=net_ring0 -l 0\n", argv[0]);
        return -ENODEV;
    }

    /* Creates a new mempool in memory to hold the mbufs */
    int socket_id = rte_socket_id();
    printf("Creating mempool on socket %d\n", socket_id);
    
    /* Check for available hugepages */
    printf("Hugepage info:\n");
    system("grep -i huge /proc/meminfo");
    
    /* Reduce mempool size if we're having allocation issues */
    uint32_t num_mbufs = 2048;  /* Reduced from the large NUM_MBUFS value */
    uint32_t cache_size = 128;  /* Reduced cache size */
    
    /* Try different parameters to create the mempool */
    printf("Attempting to create mempool with %u mbufs, cache size %u\n", 
           num_mbufs, cache_size);
    
    g_mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", num_mbufs,
        cache_size, 0, RTE_MBUF_DEFAULT_BUF_SIZE, socket_id);

    if (g_mbuf_pool == NULL) {
        /* Try without cache */
        printf("Cannot create mbuf pool with cache, trying without cache\n");
        g_mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", num_mbufs,
            0, 0, RTE_MBUF_DEFAULT_BUF_SIZE, socket_id);
        
        if (g_mbuf_pool == NULL) {
            /* Try with SOCKET_ID_ANY as fallback */
            printf("Cannot create mbuf pool on socket %d, trying with SOCKET_ID_ANY\n", socket_id);
            g_mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", num_mbufs,
                0, 0, RTE_MBUF_DEFAULT_BUF_SIZE, SOCKET_ID_ANY);
                
            if (g_mbuf_pool == NULL) {
                /* Final fallback: minimal configuration */
                printf("Still failing. Trying with minimal configuration\n");
                g_mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", 1024,
                    0, 0, 1024, SOCKET_ID_ANY);
                    
                if (g_mbuf_pool == NULL) {
                    printf("Cannot create mbuf pool: %s (errno: %d)\n", 
                           rte_strerror(rte_errno), rte_errno);
                    printf("Please ensure hugepages are properly configured\n");
                    return -ENOMEM;
                }
            }
        }
    }
    
    printf("Mempool created successfully at %p\n", (void*)g_mbuf_pool);

    /* Initialize all ports */
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        if (port_init(portid, g_mbuf_pool) != 0) {
            printf("Cannot init port %"PRIu16 "\n", portid);
            return -1;
        }
    }

    g_initialized = true;
    printf("DPDK initialized successfully\n");
    return 0;
}

uint16_t dpdk_get_port_count(void)
{
    return g_nb_ports;
}

int dpdk_poll(dpdk_packet_t *pkts, uint16_t max_pkts, uint64_t *bytes_received)
{
    if (!g_initialized) {
        return -ENODEV;
    }

    if (pkts == NULL || max_pkts == 0) {
        return -EINVAL;
    }

    /* Performance optimization: Use a much larger burst size */
    #define OPTIMAL_BURST_SIZE 256
    
    /* Static allocation of mbufs array for better performance */
    static struct rte_mbuf *bufs[OPTIMAL_BURST_SIZE];
    
    uint16_t port;
    int total_pkts = 0;
    uint64_t total_bytes = 0;

    /* Poll each port */
    RTE_ETH_FOREACH_DEV(port) {
        uint16_t remaining = max_pkts - total_pkts;
        if (remaining == 0) {
            break; /* No more space for packets */
        }

        /* Get as many packets as possible in one burst */
        uint16_t burst_size = RTE_MIN(RTE_MIN(remaining, OPTIMAL_BURST_SIZE), g_config.burst_size);
        uint16_t nb_rx = rte_eth_rx_burst(port, 0, bufs, burst_size);
        
        if (nb_rx == 0) {
            continue;
        }

        /* Process received packets with minimal operations */
        for (uint16_t i = 0; i < nb_rx; i++) {
            struct rte_mbuf *m = bufs[i];
            
            /* Just store pointer and length - minimal work */
            pkts[total_pkts].port = port;
            pkts[total_pkts].data = m;
            pkts[total_pkts].length = rte_pktmbuf_pkt_len(m);
            
            total_bytes += pkts[total_pkts].length;
            total_pkts++;
        }
    }

    /* Store the total bytes if requested */
    if (bytes_received) {
        *bytes_received = total_bytes;
    }

    return total_pkts;
}

void dpdk_free_packet(dpdk_packet_t *pkt)
{
    if (pkt && pkt->data) {
        rte_pktmbuf_free((struct rte_mbuf *)pkt->data);
        pkt->data = NULL;
    }
}

void dpdk_free_packets(dpdk_packet_t *pkts, int count)
{
    if (pkts == NULL || count <= 0) {
        return;
    }

    /* Optimization: directly free multiple mbufs at once */
    struct rte_mbuf *mbufs[64]; /* Process in chunks of 64 */
    int remaining = count;
    int offset = 0;
    
    while (remaining > 0) {
        int chunk_size = RTE_MIN(remaining, 64);
        
        /* Collect mbufs pointers */
        for (int i = 0; i < chunk_size; i++) {
            mbufs[i] = (struct rte_mbuf *)pkts[offset + i].data;
            pkts[offset + i].data = NULL;
        }
        
        /* Free all mbufs in this chunk at once */
        for (int i = 0; i < chunk_size; i++) {
            rte_pktmbuf_free(mbufs[i]);
        }
        
        remaining -= chunk_size;
        offset += chunk_size;
    }
}

void dpdk_cleanup(void)
{
    if (!g_initialized) {
        return;
    }

    printf("Cleaning up DPDK...\n");
    
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        printf("Closing port %d...\n", portid);
        rte_eth_dev_stop(portid);
        rte_eth_dev_close(portid);
    }

    /* Clean up the EAL */
    rte_eal_cleanup();

    g_initialized = false;
    g_nb_ports = 0;
    g_mbuf_pool = NULL;
    
    printf("DPDK cleanup complete\n");
}

/* Get current timestamp */
static uint64_t get_timestamp_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec;
}

/* Get current timestamp in microseconds */
static uint64_t get_timestamp_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
}

/* Metrics API implementation */

int dpdk_metrics_init(dpdk_metrics_t *metrics, uint16_t num_ports)
{
    if (!metrics || num_ports == 0) {
        return -EINVAL;
    }

    /* Clear the metrics structure */
    memset(metrics, 0, sizeof(dpdk_metrics_t));
    
    /* Allocate memory for per-port metrics */
    metrics->ports = calloc(num_ports, sizeof(dpdk_port_metrics_t));
    if (!metrics->ports) {
        return -ENOMEM;
    }
    
    /* Set the start time */
    metrics->start_time_sec = get_timestamp_sec();
    
    printf("Metrics collection initialized for %u ports\n", num_ports);
    return 0;
}

void dpdk_metrics_update(dpdk_metrics_t *metrics, uint16_t port, 
                        uint32_t rx_packets, uint64_t rx_bytes,
                        uint32_t processed_packets, uint64_t processing_time_us)
{
    if (!metrics || !metrics->ports) {
        return;
    }
    
    /* Update port-specific metrics */
    metrics->ports[port].rx_packets += rx_packets;
    metrics->ports[port].rx_bytes += rx_bytes;
    metrics->ports[port].processed_packets += processed_packets;
    metrics->ports[port].processing_time_us += processing_time_us;
    
    /* Update global metrics */
    metrics->total_rx_packets += rx_packets;
    metrics->total_rx_bytes += rx_bytes;
    metrics->total_processed_packets += processed_packets;
    metrics->total_processing_time_us += processing_time_us;
}

void dpdk_metrics_print(dpdk_metrics_t *metrics)
{
    if (!metrics || !metrics->ports) {
        printf("No metrics available\n");
        return;
    }
    
    uint64_t current_time = get_timestamp_sec();
    uint64_t runtime_sec = current_time - metrics->start_time_sec;
    uint16_t num_ports = dpdk_get_port_count();
    
    printf("\n================================================================================\n");
    printf("DPDK PACKET PROCESSING METRICS\n");
    printf("================================================================================\n");
    printf("Runtime: %"PRIu64" seconds\n", runtime_sec);
    printf("Total RX Packets: %"PRIu64"\n", metrics->total_rx_packets);
    printf("Total RX Bytes: %"PRIu64" (%.2f MB)\n", 
           metrics->total_rx_bytes, 
           metrics->total_rx_bytes / (1024.0 * 1024.0));
    printf("Total Processed Packets: %"PRIu64"\n", metrics->total_processed_packets);
    
    if (runtime_sec > 0) {
        printf("Average RX Rate: %.2f pps, %.2f Mbps\n",
               (double)metrics->total_rx_packets / runtime_sec,
               (double)metrics->total_rx_bytes * 8 / runtime_sec / (1024*1024));
        
        printf("Average Processing Rate: %.2f pps\n",
               (double)metrics->total_processed_packets / runtime_sec);
    }
    
    if (metrics->total_processed_packets > 0) {
        printf("Average Processing Time: %.2f us per packet\n",
               (double)metrics->total_processing_time_us / metrics->total_processed_packets);
    }
    
    printf("\nPER-PORT STATISTICS:\n");
    printf("%-5s %-12s %-12s %-12s %-15s\n",
           "Port", "RX Packets", "RX Bytes", "Processed", "Avg Time (us)");
    printf("--------------------------------------------------------------------------------\n");
    
    for (uint16_t port = 0; port < num_ports; port++) {
        double avg_time = 0;
        if (metrics->ports[port].processed_packets > 0) {
            avg_time = (double)metrics->ports[port].processing_time_us / 
                      metrics->ports[port].processed_packets;
        }
        
        printf("%-5u %-12"PRIu64" %-12"PRIu64" %-12"PRIu64" %-15.2f\n",
               port, 
               metrics->ports[port].rx_packets, 
               metrics->ports[port].rx_bytes,
               metrics->ports[port].processed_packets,
               avg_time);
    }
    printf("================================================================================\n");
}

void dpdk_metrics_cleanup(dpdk_metrics_t *metrics)
{
    if (metrics && metrics->ports) {
        free(metrics->ports);
        metrics->ports = NULL;
    }
} 
#!/bin/bash
# Script to reset and reconfigure hugepages for DPDK

set -e

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo $0)"
  exit 1
fi

echo "=== Resetting and reconfiguring hugepages ==="

# Check if hugepage mount exists, create if not
if [ ! -d /mnt/huge ]; then
    mkdir -p /mnt/huge
    echo "Created /mnt/huge directory"
fi

# Unmount if already mounted
if mount | grep -q "hugetlbfs"; then
    umount /mnt/huge
    echo "Unmounted existing hugepage filesystem"
fi

# Mount hugepages
mount -t hugetlbfs nodev /mnt/huge
echo "Mounted hugepage filesystem"

# Get number of NUMA nodes
NUM_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
if [ -z "$NUM_NODES" ] || [ "$NUM_NODES" -eq 0 ]; then
    NUM_NODES=1
fi

echo "Detected $NUM_NODES NUMA node(s)"

# Reset and reconfigure hugepages for each NUMA node
for ((node=0; node<$NUM_NODES; node++)); do
    node_path="/sys/devices/system/node/node$node/hugepages/hugepages-2048kB/nr_hugepages"
    if [ -f "$node_path" ]; then
        # Reset hugepages to 0
        echo 0 > $node_path
        echo "Reset hugepages for NUMA node $node"
        
        # Configure new hugepages
        echo 256 > $node_path
        current=$(cat $node_path)
        echo "Configured $current hugepages for NUMA node $node"
    else
        echo "Warning: Could not configure hugepages for NUMA node $node (path not found)"
    fi
done

# Verify configuration
echo ""
echo "Hugepage status:"
grep -i huge /proc/meminfo

echo ""
echo "=== Hugepage reconfiguration complete ==="
echo "You should now be able to run DPDK applications" 
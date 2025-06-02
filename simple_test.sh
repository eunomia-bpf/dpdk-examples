#!/bin/bash

# DPDK Test and Benchmark Framework

set -e

echo "=== DPDK Test and Benchmark Framework ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo $0)"
  exit 1
fi

# Check if DPDK GPU support is available
HAS_GPU_SUPPORT=0
if [ -f "/usr/include/rte_gpu.h" ] || [ -f "/usr/local/include/rte_gpu.h" ]; then
    HAS_GPU_SUPPORT=1
    echo "DPDK GPU support detected."
else
    echo "DPDK GPU support NOT detected. Only CPU tests will be run."
    echo "To install DPDK with GPU support, run: sudo make install-deps"
fi

# Reset and setup hugepages function
reset_and_setup_hugepages() {
    echo "Resetting and setting up hugepages..."
    
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
    echo "Hugepage status:"
    grep -i huge /proc/meminfo
}

# Compile the applications
echo "1. Compiling the DPDK applications..."
make

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
else
    echo "✗ Compilation failed!"
    exit 1
fi

# Reset and setup hugepages
echo "2. Resetting and setting up hugepages..."
reset_and_setup_hugepages

# Run tests and benchmarks
TEST_DURATION=30  # seconds

run_test() {
    local app=$1
    local test_type=$2
    local duration=$3
    
    echo ""
    echo "=== Running $app with $test_type (${duration}s) ==="
    
    # Check if the application exists in current directory
    if [ ! -f "./$app" ]; then
        # Check if application exists in eBPF-on-GPU/example directory
        if [ -f "/root/yunwei37/eBPF-on-GPU/example/$app" ]; then
            echo "Using $app from /root/yunwei37/eBPF-on-GPU/example/"
            app="/root/yunwei37/eBPF-on-GPU/example/$app"
        else
            echo "❌ $app not found. Skipping test."
            return 1
        fi
    fi
    
    if [ "$test_type" = "null" ]; then
        # Run with null PMD (auto-generates packets)
        timeout $duration $app --vdev=net_null0 -l 0 --file-prefix=$app$$ -m 256 || echo "Test completed"
        
    elif [ "$test_type" = "tap" ]; then
        # Create and configure the TAP interface before starting the DPDK application
        echo "Creating TAP interface 'test0'..."
        ip tuntap add dev test0 mode tap
        if [ $? -ne 0 ]; then
            echo "✗ Failed to create TAP interface. It might already exist or you may not have permissions."
            ip link delete test0 2>/dev/null || true
            ip tuntap add dev test0 mode tap
        fi
        
        ip link set test0 up
        ip addr add 192.168.100.1/24 dev test0
        
        if ip link show test0 | grep -q "UP"; then
            echo "✓ TAP interface 'test0' created and configured successfully"
            
            # Start the DPDK application with the TAP interface
            $app --vdev=net_tap0,iface=test0 -l 0 --file-prefix=$app$$ -m 256 &
            local pid=$!
            
            echo "Generating traffic for ${duration}s..."
            
            # Start a ping process that generates constant traffic
            ping -f 192.168.100.2 > /dev/null 2>&1 &
            local ping_pid=$!
            
            # Let it run for the specified duration
            sleep $duration
            
            # Stop ping and application
            kill $ping_pid 2>/dev/null || true
            kill $pid 2>/dev/null || true
            wait $pid 2>/dev/null || true
            
            # Clean up interface
            echo "Cleaning up TAP interface..."
            ip link delete test0 2>/dev/null || true
        else
            echo "✗ Failed to configure TAP interface"
            ip link delete test0 2>/dev/null || true
        fi
    fi
    
    echo "Test completed."
    return 0
}

extract_metrics() {
    local metric_name=$1
    local output=$2
    
    local value=$(echo "$output" | grep "$metric_name" | tail -1 | awk -F ':' '{print $2}' | tr -d ' ' | tr -d 'pps' | tr -d 'Mbps')
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

echo ""
echo "3. Running tests and benchmarks..."

# Run dpdk_example tests
echo "Running CPU tests with dpdk_example..."
CPU_NULL_OUTPUT=$(run_test "dpdk_example" "null" $TEST_DURATION)
CPU_TAP_OUTPUT=$(run_test "dpdk_example" "tap" $TEST_DURATION)

# Run dpdk_gpu tests only if GPU support is available
if [ $HAS_GPU_SUPPORT -eq 1 ] && [ -f "./dpdk_gpu" ]; then
    echo "Running GPU tests with dpdk_gpu..."
    GPU_NULL_OUTPUT=$(run_test "dpdk_gpu" "null" $TEST_DURATION)
    GPU_TAP_OUTPUT=$(run_test "dpdk_gpu" "tap" $TEST_DURATION)
else
    echo ""
    echo "Skipping GPU tests (GPU support not available or dpdk_gpu not built)"
fi

# Print results summary
echo ""
echo "=== Benchmark Results Summary ==="
echo "=================================="

# Extract metrics from output
CPU_NULL_RX_RATE=$(extract_metrics "Average RX Rate" "$CPU_NULL_OUTPUT")
if [ "$CPU_NULL_RX_RATE" = "N/A" ]; then
    # Try with our simplified metrics format
    CPU_NULL_RX_RATE=$(extract_metrics "Packets per second" "$CPU_NULL_OUTPUT")
fi
echo "CPU Null PMD RX Rate: ${CPU_NULL_RX_RATE} pps"

CPU_TAP_RX_RATE=$(extract_metrics "Average RX Rate" "$CPU_TAP_OUTPUT")
if [ "$CPU_TAP_RX_RATE" = "N/A" ]; then
    # Try with our simplified metrics format
    CPU_TAP_RX_RATE=$(extract_metrics "Packets per second" "$CPU_TAP_OUTPUT")
fi
echo "CPU TAP RX Rate: ${CPU_TAP_RX_RATE} pps"

if [ $HAS_GPU_SUPPORT -eq 1 ] && [ -f "./dpdk_gpu" ]; then
    GPU_NULL_RX_RATE=$(extract_metrics "Average RX Rate" "$GPU_NULL_OUTPUT")
    GPU_NULL_PROC_RATE=$(extract_metrics "Average Processing Rate" "$GPU_NULL_OUTPUT")
    GPU_NULL_PROC_TIME=$(extract_metrics "Average Processing Time" "$GPU_NULL_OUTPUT")
    
    echo "GPU Null PMD RX Rate: ${GPU_NULL_RX_RATE} pps"
    echo "GPU Null PMD Processing Rate: ${GPU_NULL_PROC_RATE} pps"
    echo "GPU Null PMD Processing Time: ${GPU_NULL_PROC_TIME} us/packet"
    
    GPU_TAP_RX_RATE=$(extract_metrics "Average RX Rate" "$GPU_TAP_OUTPUT")
    GPU_TAP_PROC_RATE=$(extract_metrics "Average Processing Rate" "$GPU_TAP_OUTPUT")
    GPU_TAP_PROC_TIME=$(extract_metrics "Average Processing Time" "$GPU_TAP_OUTPUT")
    
    echo "GPU TAP RX Rate: ${GPU_TAP_RX_RATE} pps"
    echo "GPU TAP Processing Rate: ${GPU_TAP_PROC_RATE} pps"
    echo "GPU TAP Processing Time: ${GPU_TAP_PROC_TIME} us/packet"
fi

echo ""
echo "Test Summary:"
echo "- CPU tests show baseline performance"
if [ $HAS_GPU_SUPPORT -eq 1 ] && [ -f "./dpdk_gpu" ]; then
    echo "- GPU tests show GPU-accelerated packet processing performance"
    echo "- Higher RX Rate and Processing Rate values are better"
    echo "- Lower Processing Time values are better"
else
    echo "- GPU tests were not run (GPU support not available)"
    echo "- To enable GPU support, run: sudo make install-deps"
fi
echo ""
echo "==================================" 
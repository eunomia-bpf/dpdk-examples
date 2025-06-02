#!/bin/bash

# DPDK Test and Benchmark Framework

set -e

echo "=== DPDK Test and Benchmark Framework ==="

# Check if DPDK GPU support is available
HAS_GPU_SUPPORT=0
if [ -f "/usr/include/rte_gpu.h" ] || [ -f "/usr/local/include/rte_gpu.h" ]; then
    HAS_GPU_SUPPORT=1
    echo "DPDK GPU support detected."
else
    echo "DPDK GPU support NOT detected. Only CPU tests will be run."
    echo "To install DPDK with GPU support, run: sudo ./install_deps.sh"
fi

# Setup hugepages function
setup_hugepages() {
    echo "Setting up hugepages..."
    if [ ! -d /mnt/huge ]; then
        mkdir -p /mnt/huge
        mount -t hugetlbfs nodev /mnt/huge
    fi

    # Get number of NUMA nodes
    NUM_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
    if [ -z "$NUM_NODES" ] || [ "$NUM_NODES" -eq 0 ]; then
        NUM_NODES=1
    fi
    
    echo "Detected $NUM_NODES NUMA node(s)"
    
    # Configure hugepages for each NUMA node
    for ((node=0; node<$NUM_NODES; node++)); do
        node_path="/sys/devices/system/node/node$node/hugepages/hugepages-2048kB/nr_hugepages"
        if [ -f "$node_path" ]; then
            current=$(cat $node_path)
            if [ "$current" -eq 0 ]; then
                echo "Configuring 256 hugepages for NUMA node $node"
                echo 256 > $node_path
            else
                echo "NUMA node $node already has $current hugepages"
            fi
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

# Setup hugepages if not already done
echo "2. Setting up hugepages..."
setup_hugepages

# Run tests and benchmarks
TEST_DURATION=30  # seconds
NULL_OUTPUT="null_test_output.log"
TAP_OUTPUT="tap_test_output.log"
GPU_NULL_OUTPUT="gpu_null_test_output.log"
GPU_TAP_OUTPUT="gpu_tap_test_output.log"

run_test() {
    local app=$1
    local test_type=$2
    local duration=$3
    local output_file=$4
    
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
        timeout $duration $app --vdev=net_null0 -l 0 --file-prefix=$app$$ -m 256 > $output_file 2>&1 || echo "Test completed"
    elif [ "$test_type" = "tap" ]; then
        # Run with TAP interface
        $app --vdev=net_tap0,iface=test0 -l 0 --file-prefix=$app$$ -m 256 > $output_file 2>&1 &
        local pid=$!
        
        # Wait for interface creation
        sleep 2
        
        if ip link show test0 > /dev/null 2>&1; then
            echo "✓ TAP interface 'test0' created successfully"
            
            # Configure and send traffic to the interface
            ip link set test0 up
            ip addr add 192.168.100.1/24 dev test0
            
            echo "Generating traffic for ${duration}s..."
            
            # Start a ping process that generates constant traffic
            ping -f 192.168.100.1 > /dev/null 2>&1 &
            local ping_pid=$!
            
            # Let it run for the specified duration
            sleep $duration
            
            # Stop ping and application
            kill $ping_pid 2>/dev/null || true
            kill $pid 2>/dev/null || true
            wait $pid 2>/dev/null || true
            
            # Clean up interface
            ip link delete test0 2>/dev/null || true
        else
            echo "✗ Failed to create TAP interface"
            kill $pid 2>/dev/null || true
        fi
    fi
    
    echo "Test completed. Output saved to $output_file"
    return 0
}

extract_metrics() {
    local output_file=$1
    local metric_name=$2
    
    if [ ! -f "$output_file" ]; then
        echo "N/A"
        return
    fi
    
    local value=$(grep "$metric_name" $output_file | tail -1 | awk -F ':' '{print $2}' | tr -d ' ' | tr -d 'pps' | tr -d 'Mbps')
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

echo ""
echo "3. Running tests and benchmarks..."

# Run dpdk_example instead of dpdk_minimal
run_test "dpdk_example" "null" $TEST_DURATION $NULL_OUTPUT

# Run dpdk_example with TAP interface
run_test "dpdk_example" "tap" $TEST_DURATION $TAP_OUTPUT

# Run dpdk_gpu tests only if GPU support is available
if [ $HAS_GPU_SUPPORT -eq 1 ] && [ -f "./dpdk_gpu" ]; then
    run_test "dpdk_gpu" "null" $TEST_DURATION $GPU_NULL_OUTPUT
    run_test "dpdk_gpu" "tap" $TEST_DURATION $GPU_TAP_OUTPUT
else
    echo ""
    echo "Skipping GPU tests (GPU support not available or dpdk_gpu not built)"
fi

# Print results summary
echo ""
echo "=== Benchmark Results Summary ==="
echo "=================================="

# Extract metrics from logs
CPU_NULL_RX_RATE=$(extract_metrics $NULL_OUTPUT "Average RX Rate")
if [ "$CPU_NULL_RX_RATE" = "N/A" ]; then
    # Try with our simplified metrics format
    CPU_NULL_RX_RATE=$(extract_metrics $NULL_OUTPUT "Packets per second")
fi
echo "CPU Null PMD RX Rate: ${CPU_NULL_RX_RATE} pps"

CPU_TAP_RX_RATE=$(extract_metrics $TAP_OUTPUT "Average RX Rate")
if [ "$CPU_TAP_RX_RATE" = "N/A" ]; then
    # Try with our simplified metrics format
    CPU_TAP_RX_RATE=$(extract_metrics $TAP_OUTPUT "Packets per second")
fi
echo "CPU TAP RX Rate: ${CPU_TAP_RX_RATE} pps"

if [ $HAS_GPU_SUPPORT -eq 1 ] && [ -f "./dpdk_gpu" ]; then
    GPU_NULL_RX_RATE=$(extract_metrics $GPU_NULL_OUTPUT "Average RX Rate")
    GPU_NULL_PROC_RATE=$(extract_metrics $GPU_NULL_OUTPUT "Average Processing Rate")
    GPU_NULL_PROC_TIME=$(extract_metrics $GPU_NULL_OUTPUT "Average Processing Time")
    
    echo "GPU Null PMD RX Rate: ${GPU_NULL_RX_RATE} pps"
    echo "GPU Null PMD Processing Rate: ${GPU_NULL_PROC_RATE} pps"
    echo "GPU Null PMD Processing Time: ${GPU_NULL_PROC_TIME} us/packet"
    
    GPU_TAP_RX_RATE=$(extract_metrics $GPU_TAP_OUTPUT "Average RX Rate")
    GPU_TAP_PROC_RATE=$(extract_metrics $GPU_TAP_OUTPUT "Average Processing Rate")
    GPU_TAP_PROC_TIME=$(extract_metrics $GPU_TAP_OUTPUT "Average Processing Time")
    
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
    echo "- To enable GPU support, run: sudo ./install_deps.sh"
fi
echo ""
echo "All test logs are available in the output files for detailed analysis."
echo "==================================" 
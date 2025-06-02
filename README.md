# DPDK Packet Processing with GPU Acceleration

This project demonstrates DPDK packet processing with and without GPU acceleration. It includes both a minimal CPU-only version and a GPU-accelerated version for comparison.

## Features

- CPU-only and GPU-accelerated packet processing
- Works with DPDK virtual devices (null PMD, TAP)
- Performance metrics collection and benchmarking
- No physical NICs required for testing

## Prerequisites

### System Requirements

- Ubuntu 18.04/20.04/22.04 or compatible Linux distribution
- CUDA-capable GPU (for GPU acceleration)
- DPDK 20.11 or later with GPU support
- CUDA Toolkit 10.0 or later

## Installation

### Quick Start

The simplest way to get started is to run:

```bash
# Clone the repository
git clone <repository-url>
cd dpdk-examples

# Install DPDK with GPU support (if needed)
sudo ./install_deps.sh

# Build the applications
make

# Run the benchmark suite
sudo ./simple_test.sh
```

### Manual Installation

#### 1. Install DPDK with GPU Support

DPDK with GPU support requires a specific build configuration. Here's how to install it:

```bash
# Install basic dependencies
sudo apt update
sudo apt install -y build-essential meson ninja-build python3-pyelftools libnuma-dev

# Install CUDA Toolkit (if not already installed)
# Visit https://developer.nvidia.com/cuda-downloads for the latest version
# Example for Ubuntu 20.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda

# Download and build DPDK with GPU support
git clone https://github.com/DPDK/dpdk.git
cd dpdk
git checkout v21.11 # Or a later version
meson -Denable_docs=false -Denable_kmods=false -Dlibdir=lib -Denable_drivers=net/null,net/tap,gpu/cuda builddir
cd builddir
ninja
sudo ninja install
sudo ldconfig

# Verify installation
pkg-config --modversion libdpdk
```

#### 2. Configure Hugepages

DPDK requires hugepages for its memory pools. You can configure them using:

```bash
# Configure hugepages on all NUMA nodes
sudo ./reset_hugepages.sh
```

Or manually:

```bash
# Create mount point if it doesn't exist
sudo mkdir -p /mnt/huge

# Mount hugetlbfs
sudo mount -t hugetlbfs nodev /mnt/huge

# Configure 256 hugepages for each NUMA node
# For a single NUMA node system:
echo 256 | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages

# For multi-NUMA node systems, repeat for each node:
echo 256 | sudo tee /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
```

## Usage

### Basic Testing

```bash
# Test CPU-only version with null PMD (auto-generates packets)
sudo make test-null

# Test CPU-only version with TAP interface
sudo make test-tap

# Test GPU-accelerated version with null PMD
sudo make test-gpu-null

# Test GPU-accelerated version with TAP interface
sudo make test-gpu-tap
```

### Run Benchmark Suite

```bash
# Run comprehensive benchmarks for both CPU and GPU versions
sudo make benchmark

# Or directly:
sudo ./simple_test.sh
```

## Project Structure

- `dpdk_driver.h/c` - Core DPDK functionality and GPU integration
- `dpdk_minimal.c` - CPU-only packet processing application
- `dpdk_gpu.c` - GPU-accelerated packet processing application
- `simple_test.sh` - Automated test and benchmark script
- `install_deps.sh` - Script to install DPDK with GPU support
- `reset_hugepages.sh` - Script to reset and reconfigure hugepages

## Benchmarking

The benchmark suite measures and compares:

- Packet throughput (packets per second)
- Processing time per packet
- Total processed packets
- CPU vs. GPU performance

Example benchmark output:

```
=== Benchmark Results Summary ===
==================================
CPU Null PMD RX Rate: 1245678.45 pps
CPU TAP RX Rate: 23456.78 pps
GPU Null PMD RX Rate: 2345678.90 pps
GPU Null PMD Processing Rate: 2345678.90 pps
GPU Null PMD Processing Time: 0.42 us/packet
GPU TAP RX Rate: 45678.90 pps
GPU TAP Processing Rate: 45678.90 pps
GPU TAP Processing Time: 0.45 us/packet
```

## Understanding the Code

### CPU Version (`dpdk_minimal.c`)

The CPU-only version demonstrates:
- DPDK initialization and setup
- Packet reception and processing
- Metrics collection

### GPU Version (`dpdk_gpu.c`)

The GPU-accelerated version adds:
- CPU-GPU communication
- CUDA kernel for parallel packet processing
- Performance optimization with GPU memory

### GPU Processing Flow

1. CPU receives packets via DPDK
2. Packets are sent to GPU via DPDK-GPU communication API
3. CUDA kernel processes packets in parallel
4. Results are synchronized back to CPU
5. Performance metrics are collected and reported

## Troubleshooting

### Hugepage Issues

If you see errors like:
```
EAL: No free 2048 kB hugepages reported on node X
MBUF: error setting mempool handler
Cannot create mbuf pool
```

Reset your hugepages using:
```
sudo ./reset_hugepages.sh
```

This script will:
1. Unmount and remount the hugepage filesystem
2. Reset and reconfigure hugepages for all NUMA nodes
3. Verify the configuration

### Missing DPDK GPU Support

If you see errors like `rte_gpu.h: No such file or directory`, your DPDK installation doesn't include GPU support. Follow the installation instructions above to build DPDK with GPU support.

If GPU support is not available, the build system will automatically disable GPU features and only build the CPU-only version.

### CUDA Issues

- Ensure CUDA Toolkit is installed: `nvcc --version`
- Check GPU availability: `nvidia-smi`
- Update CUDA_HOME in Makefile if needed

### Common DPDK Issues

- **Permission denied**: Run with `sudo` or add user to appropriate groups
- **No ports found**: Make sure to use `--vdev` parameter
- **Multiple NUMA nodes**: Use `reset_hugepages.sh` to configure all nodes properly

## Learn More

- [DPDK Documentation](https://doc.dpdk.org/)
- [DPDK GPU Guide](https://doc.dpdk.org/guides/prog_guide/gpu_accel.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 
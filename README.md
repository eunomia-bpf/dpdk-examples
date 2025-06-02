# DPDK GPU Packet Processing

This application demonstrates how to use DPDK with GPU acceleration for packet processing.

## Prerequisites

- DPDK (version 21.11 or later)
- CUDA Toolkit (version 10.0 or later)
- Linux with supported NICs for DPDK

## Building the Application

1. Make sure CUDA is installed and `nvcc` is in your PATH
2. Set up the DPDK environment variables (if not already done)
3. Make the build script executable and run it:

```bash
chmod +x build_dpdk_gpu.sh
./build_dpdk_gpu.sh
```

The application will be built in the `build` directory.

## Running the Application

The application requires DPDK EAL initialization parameters. You can run it with:

```bash
sudo ./build/dpdk_gpu [EAL options]
```

For example:

```bash
# Run with 4 cores on socket 0 and use a virtual device
sudo ./build/dpdk_gpu -l 0-3 -n 4 --vdev=net_pcap0,iface=eth0
```

### Notes on GPU Processing

This application demonstrates:

1. Using the DPDK GPU device library (rte_gpudev.h) to integrate with CUDA
2. Processing packets on the GPU with zero CPU fallback
3. Efficient communication between CPU and GPU using DPDK's GPU communication API

If CUDA is not available during compilation, the application will fail to build, as GPU processing is required.

## Implementation Details

- The application uses DPDK's communication flags and lists to coordinate between CPU and GPU
- Packets are received by the CPU and passed to the GPU for processing
- The GPU increments the first byte of each packet as a simple example
- A warning is shown if packets are dropped because all GPU processing slots are busy

## Troubleshooting

- If the build fails with missing DPDK libraries, ensure your RTE_SDK is set correctly
- If there are CUDA errors, check that the CUDA Toolkit is installed correctly
- To see more detailed logs, run with the EAL log level option: `--log-level=8` 
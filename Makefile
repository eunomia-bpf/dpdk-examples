# DPDK Examples Makefile

# Binary names
MINIMAL_APP = dpdk_minimal
GPU_APP = dpdk_gpu

# Source files
COMMON_SRCS = dpdk_driver.c
MINIMAL_SRCS = dpdk_minimal.c
GPU_SRCS = dpdk_gpu.c
GPU_CUDA_SRCS = dpdk_gpu_kernel.cu

# Object files
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
MINIMAL_OBJS = $(MINIMAL_SRCS:.c=.o)
GPU_OBJS = $(GPU_SRCS:.c=.o)
GPU_CUDA_OBJS = $(GPU_CUDA_SRCS:.cu=.o)

# Header files
HEADERS = dpdk_driver.h dpdk_gpu.h

# DPDK configuration
DPDK_CFLAGS = $(shell pkg-config --cflags libdpdk)
DPDK_LDFLAGS = $(shell pkg-config --libs libdpdk)

# CUDA configuration
CUDA_CFLAGS = -I/usr/include -I/usr/local/cuda/include
CUDA_LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64 -lcudart -lcuda

# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -D_GNU_SOURCE -include rte_config.h $(DPDK_CFLAGS)
LDFLAGS = $(DPDK_LDFLAGS) -lrt -lpthread -ldl
# For NVCC, we need to pass library flags differently
NVCC_LDFLAGS = -Xlinker -L/usr/local/lib $(shell pkg-config --libs-only-l libdpdk | sed 's/-l/\-Xlinker -l/g') -Xlinker -lrt -Xlinker -lpthread -Xlinker -ldl

# Default target builds both apps
all: $(MINIMAL_APP) $(GPU_APP)

# Build the minimal application
$(MINIMAL_APP): $(COMMON_OBJS) $(MINIMAL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build the GPU application
$(GPU_APP): $(COMMON_OBJS) $(GPU_OBJS) $(GPU_CUDA_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

# Build GPU object file
$(GPU_OBJS): $(GPU_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -c -o $@ $<

# Build CUDA object file with nvcc
$(GPU_CUDA_OBJS): $(GPU_CUDA_SRCS)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

# Build regular object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean rule
clean:
	rm -f $(MINIMAL_APP) $(GPU_APP) $(COMMON_OBJS) $(MINIMAL_OBJS) $(GPU_OBJS) $(GPU_CUDA_OBJS)

# Test with different virtual devices
test-null:
	sudo ./$(MINIMAL_APP) --vdev=net_null0 -l 0

test-tap:
	sudo ./$(MINIMAL_APP) --vdev=net_tap0,iface=test0 -l 0

test-gpu-null:
	sudo ./$(GPU_APP) --vdev=net_null0 -l 0

test-gpu-tap:
	sudo ./$(GPU_APP) --vdev=net_tap0,iface=test0 -l 0

# Setup hugepages needed for DPDK
setup-hugepages:
	mkdir -p /mnt/huge
	mount -t hugetlbfs nodev /mnt/huge
	echo 256 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build all applications"
	@echo "  $(MINIMAL_APP)       - Build minimal DPDK application"
	@echo "  $(GPU_APP)          - Build GPU-accelerated DPDK application"
	@echo "  clean         - Clean build artifacts"
	@echo "  test-null     - Test minimal app with null PMD"
	@echo "  test-tap      - Test minimal app with TAP interface"
	@echo "  test-gpu-null - Test GPU app with null PMD"
	@echo "  test-gpu-tap  - Test GPU app with TAP interface"
	@echo "  setup-hugepages - Setup hugepages required for DPDK"
	@echo "  help          - Show this help"

.PHONY: all clean test-null test-tap test-gpu-null test-gpu-tap setup-hugepages help 
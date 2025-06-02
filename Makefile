# DPDK Examples Makefile

# Binary names
MINIMAL_APP = dpdk_minimal
GPU_APP = dpdk_gpu

# Common source files
COMMON_SRCS = dpdk_driver.c

# App-specific source files
MINIMAL_SRCS = dpdk_minimal.c
GPU_SRCS = dpdk_gpu.c

# Object files
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
MINIMAL_OBJS = $(MINIMAL_SRCS:.c=.o)
GPU_OBJS = $(GPU_SRCS:.c=.o)

# Header files
HEADERS = dpdk_driver.h

# DPDK configuration
DPDK_CFLAGS = $(shell pkg-config --cflags libdpdk)
DPDK_LDFLAGS = $(shell pkg-config --libs libdpdk)

# CUDA configuration
CUDA_HOME ?= /usr/local/cuda
CUDA_CFLAGS = -I$(CUDA_HOME)/include
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda

# Check if GPU support is available
HAS_GPU_SUPPORT = 0
GPU_HEADER = /usr/include/rte_gpu.h
LOCAL_GPU_HEADER = /usr/local/include/rte_gpu.h
ifneq ("$(wildcard $(GPU_HEADER))","")
    HAS_GPU_SUPPORT = 1
endif
ifneq ("$(wildcard $(LOCAL_GPU_HEADER))","")
    HAS_GPU_SUPPORT = 1
endif

# Define GPU support flag if available
ifeq ($(HAS_GPU_SUPPORT), 1)
    GPU_CFLAGS = -DRTE_HAS_GPU_SUPPORT
else
    $(warning "DPDK GPU support not detected. Building without GPU acceleration.")
endif

# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -Wextra -std=c99 -D_GNU_SOURCE -include rte_config.h $(DPDK_CFLAGS) $(GPU_CFLAGS)
LDFLAGS = $(DPDK_LDFLAGS) -lrt -lpthread -ldl

# Default target builds minimal application (always works)
all: $(MINIMAL_APP)

# If GPU support is available, build GPU app too
ifeq ($(HAS_GPU_SUPPORT), 1)
all: $(GPU_APP)
endif

# Build the minimal application
$(MINIMAL_APP): $(COMMON_OBJS) $(MINIMAL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build the GPU application (only with GPU support)
ifeq ($(HAS_GPU_SUPPORT), 1)
$(GPU_APP): $(COMMON_OBJS) $(GPU_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

# Build GPU object file with nvcc
$(GPU_OBJS): $(GPU_SRCS)
	$(NVCC) -x cu --compiler-options "-fPIC $(CFLAGS) $(CUDA_CFLAGS)" -dc -o $@ $<
endif

# Build regular object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: clean all

# Clean rule
clean:
	rm -f $(MINIMAL_APP) $(GPU_APP) $(COMMON_OBJS) $(MINIMAL_OBJS) $(GPU_OBJS)

# Test with different virtual devices
test-null:
	sudo ./$(MINIMAL_APP) --vdev=net_null0 -l 0

test-tap:
	sudo ./$(MINIMAL_APP) --vdev=net_tap0,iface=test0 -l 0

# GPU tests - only run if GPU app is built
test-gpu-null: check-gpu-app
	sudo ./$(GPU_APP) --vdev=net_null0 -l 0

test-gpu-tap: check-gpu-app
	sudo ./$(GPU_APP) --vdev=net_tap0,iface=test0 -l 0

# Check if GPU app exists
check-gpu-app:
ifeq ($(HAS_GPU_SUPPORT), 0)
	@echo "GPU support not available. Cannot run GPU tests."
	@exit 1
endif
ifeq ($(wildcard $(GPU_APP)),)
	@echo "GPU application not built. Run 'make $(GPU_APP)' first."
	@exit 1
endif

# Run full benchmark suite
benchmark:
	sudo ./simple_test.sh

# Show DPDK configuration
dpdk-info:
	@echo "DPDK CFLAGS: $(DPDK_CFLAGS)"
	@echo "DPDK LDFLAGS: $(DPDK_LDFLAGS)"
	@echo "GPU Support: $(HAS_GPU_SUPPORT)"
	@pkg-config --modversion libdpdk

# Install DPDK with GPU support
install-deps:
	sudo ./install_deps.sh

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build available applications (always includes minimal)"
	@echo "  $(MINIMAL_APP)       - Build minimal DPDK application"
ifeq ($(HAS_GPU_SUPPORT), 1)
	@echo "  $(GPU_APP)          - Build GPU-accelerated DPDK application"
endif
	@echo "  debug         - Build debug versions"
	@echo "  clean         - Clean build artifacts"
	@echo "  test-null     - Test minimal app with null PMD"
	@echo "  test-tap      - Test minimal app with TAP interface"
ifeq ($(HAS_GPU_SUPPORT), 1)
	@echo "  test-gpu-null - Test GPU app with null PMD"
	@echo "  test-gpu-tap  - Test GPU app with TAP interface"
endif
	@echo "  benchmark     - Run full benchmark suite"
	@echo "  dpdk-info     - Show DPDK configuration"
	@echo "  install-deps  - Install DPDK with GPU support"
	@echo "  help          - Show this help"
	@echo ""
	@echo "GPU Support: $(HAS_GPU_SUPPORT)"
ifeq ($(HAS_GPU_SUPPORT), 0)
	@echo "To enable GPU support, run 'make install-deps' to install DPDK with GPU support"
endif

.PHONY: all debug clean test-null test-tap test-gpu-null test-gpu-tap benchmark dpdk-info install-deps help check-gpu-app 
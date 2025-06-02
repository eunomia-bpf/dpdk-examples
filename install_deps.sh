#!/bin/bash
# DPDK with GPU Support Installation Script

set -e

echo "=== DPDK with GPU Support Installation Script ==="
echo "This script will install DPDK with GPU/CUDA support"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo $0)"
  exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Cannot detect OS, assuming Ubuntu"
    OS="Ubuntu"
    VER="20.04"
fi

echo "Detected OS: $OS $VER"

# Install basic dependencies
echo "Installing basic dependencies..."
apt update
apt install -y build-essential meson ninja-build python3-pyelftools libnuma-dev pkg-config git

# Check if CUDA is installed
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA Toolkit already installed (version $CUDA_VERSION)"
else
    echo "CUDA Toolkit not found. Please install it manually."
    echo "Visit https://developer.nvidia.com/cuda-downloads for instructions."
    echo "Example for Ubuntu 20.04:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
    echo "  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
    echo "  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub"
    echo "  sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /\""
    echo "  sudo apt update"
    echo "  sudo apt install -y cuda"
    
    echo ""
    echo "This script will continue to install DPDK with GPU support,"
    echo "but you need to install CUDA manually for GPU acceleration to work."
    
    read -p "Continue without CUDA? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for existing DPDK installation
if pkg-config --exists libdpdk; then
    DPDK_VERSION=$(pkg-config --modversion libdpdk)
    echo "DPDK already installed (version $DPDK_VERSION)"
    
    # Check for GPU support
    if [ -f "/usr/local/include/rte_gpu.h" ] || [ -f "/usr/include/rte_gpu.h" ]; then
        echo "DPDK GPU support appears to be installed."
        
        read -p "Reinstall DPDK with GPU support? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping DPDK installation."
            
            # Setup hugepages (required for DPDK)
            setup_hugepages
            
            exit 0
        fi
    else
        echo "Existing DPDK installation does not have GPU support."
    fi
fi

# Setup hugepages function
setup_hugepages() {
    echo "Setting up hugepages..."
    if [ ! -d /mnt/huge ]; then
        mkdir -p /mnt/huge
        mount -t hugetlbfs nodev /mnt/huge
        
        # Add to fstab if not already there
        if ! grep -q "hugetlbfs" /etc/fstab; then
            echo "nodev /mnt/huge hugetlbfs defaults 0 0" >> /etc/fstab
            echo "Added hugepages mount to /etc/fstab"
        fi
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

# Download and build DPDK with GPU support
echo "Building DPDK with GPU support..."
WORK_DIR=$(mktemp -d)
cd $WORK_DIR

echo "Downloading DPDK..."
git clone https://github.com/DPDK/dpdk.git
cd dpdk

# Get the latest stable release tag
LATEST_TAG=$(git describe --tags `git rev-list --tags --max-count=1`)
echo "Latest DPDK tag: $LATEST_TAG"
git checkout $LATEST_TAG

echo "Configuring DPDK with GPU support..."
meson -Denable_docs=false -Denable_kmods=false -Dlibdir=lib -Denable_drivers=net/null,net/tap,gpu/cuda builddir
cd builddir

echo "Building DPDK (this may take a while)..."
ninja
echo "Installing DPDK..."
ninja install
ldconfig

# Verify installation
if pkg-config --exists libdpdk; then
    DPDK_VERSION=$(pkg-config --modversion libdpdk)
    echo "DPDK installation successful (version $DPDK_VERSION)"
else
    echo "DPDK installation failed!"
    exit 1
fi

# Check if GPU support was successfully installed
if [ -f "/usr/local/include/rte_gpu.h" ] || [ -f "/usr/include/rte_gpu.h" ]; then
    echo "✅ DPDK GPU support successfully installed!"
else
    echo "⚠️  DPDK installed but GPU support seems to be missing!"
    echo "   This may happen if the DPDK version doesn't support GPU properly."
    echo "   You can still use the CPU-only application."
fi

# Setup hugepages
setup_hugepages

# Cleanup
cd /

echo ""
echo "=== Installation Complete ==="
echo "DPDK with GPU support has been installed."
echo "You can now build the examples with 'make'"
echo "" 
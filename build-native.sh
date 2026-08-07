#!/usr/bin/env bash
# Build the image for THIS machine: CPU with -march=native, CUDA for exactly
# the installed GPU's compute capability.
#
#   ./build-native.sh [image-tag]        # default tag: bonsai-27b:native
#
# Why a script and not just `docker build`: with the default runc runtime the
# build container has no GPU, so CMake cannot resolve
# CMAKE_CUDA_ARCHITECTURES=native on its own. We read the compute capability
# from the host driver and hand it over.
set -euo pipefail

TAG="${1:-bonsai-27b:native}"

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi not found — an NVIDIA driver is required." >&2; exit 1; }

# "8.9" -> "89"; multiple GPUs -> all their capabilities, deduplicated
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
       | tr -d ' .' | sort -u | paste -sd';' -)

[ -n "$ARCH" ] || { echo "Could not read the compute capability." >&2; exit 1; }

echo "GPU compute capability: ${ARCH}"
echo "Building ${TAG} (CPU: -march=native, CUDA: sm_${ARCH//;/, sm_})"

exec docker build -f Dockerfile.native \
    --build-arg CUDA_DOCKER_ARCH="${ARCH}" \
    --build-arg GGML_NATIVE=ON \
    -t "${TAG}" "$(dirname "$0")"

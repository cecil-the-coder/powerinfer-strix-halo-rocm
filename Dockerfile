# PowerInfer with ROCm for AMD Strix Halo (gfx1151)
# Based on ROCm 7.1.1 optimized for Strix Halo UMA architecture
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest

# Use official ROCm dev image for building (has LLVM compiler + all dev tools)
ARG BUILD_IMAGE=rocm/dev-ubuntu-22.04:6.2
# Runtime can use the lighter Strix Halo optimized image
ARG RUNTIME_IMAGE=kyuz0/amd-strix-halo-toolboxes:rocm-7.1.1

FROM ${BUILD_IMAGE} AS builder

# Build arguments
ARG AMDGPU_TARGETS=gfx1151
ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main

# Environment for ROCm build
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    GPU_TARGETS=${AMDGPU_TARGETS} \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
    CMAKE_PREFIX_PATH=/opt/rocm \
    PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone PowerInfer
RUN git clone --depth 1 --branch ${POWERINFER_BRANCH} ${POWERINFER_REPO} powerinfer

WORKDIR /build/powerinfer

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Build PowerInfer with ROCm/HIP support
# Per PowerInfer docs: https://github.com/SJTU-IPADS/PowerInfer#amd-gpus
# Key flags:
#   LLAMA_HIPBLAS=ON     - Enable HIP BLAS for ROCm
#   AMDGPU_TARGETS       - Target gfx1151 specifically (Strix Halo)
#   CMAKE_PREFIX_PATH    - Help CMake find ROCm libraries
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DLLAMA_HIPBLAS=ON \
    -DAMDGPU_TARGETS=${AMDGPU_TARGETS} \
    && cmake --build build --config Release -j$(nproc)

# Stage artifacts for runtime image
RUN mkdir -p /staging/bin /staging/lib \
    && cp -r build/bin/* /staging/bin/ 2>/dev/null || true \
    && find build -name "*.so" -exec cp {} /staging/lib/ \; 2>/dev/null || true \
    && ls -la /staging/bin /staging/lib

# Runtime stage - smaller image using Strix Halo optimized image
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS runtime

# Runtime environment
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=gfx1151 \
    GPU_TARGETS=gfx1151 \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH="/opt/rocm/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/app"

# Copy staged binaries and libraries
COPY --from=builder /staging/bin/ /app/
COPY --from=builder /staging/lib/ /app/

WORKDIR /app

# Create models directory
RUN mkdir -p /models

# Default command - show help
CMD ["./main", "--help"]

# Labels
LABEL maintainer="PowerInfer ROCm Build" \
      description="PowerInfer with ROCm support for AMD Strix Halo (gfx1151)" \
      rocm.version="7.1.1" \
      gpu.target="gfx1151"

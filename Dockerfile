# PowerInfer with ROCm for AMD Strix Halo (gfx1151)
# Based on ROCm 7.1.1 optimized for Strix Halo UMA architecture
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest

# Use official ROCm dev image for building (has LLVM compiler + all dev tools)
ARG BUILD_IMAGE=rocm/dev-ubuntu-22.04:6.2
# Use same ROCm image for runtime to ensure library compatibility
ARG RUNTIME_IMAGE=rocm/dev-ubuntu-22.04:6.2

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

# Install build dependencies including ROCm hipBLAS/rocBLAS dev packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    wget \
    curl \
    hipblas-dev \
    rocblas-dev \
    rocsolver-dev \
    rocsparse-dev \
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

# Verify HIP is available before building
RUN hipcc --version && echo "HIP compiler found"

# Configure with explicit HIP paths
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
    -DLLAMA_HIPBLAS=ON \
    -DAMDGPU_TARGETS=${AMDGPU_TARGETS} \
    -DGPU_TARGETS=${AMDGPU_TARGETS} \
    2>&1 | tee cmake_config.log \
    && grep -i "hipblas\|hip\|rocm" cmake_config.log || true

# Build and verify HIP was actually used
RUN cmake --build build --config Release -j$(nproc) \
    && ldd build/bin/main | grep -i hip && echo "HIP libraries linked successfully" \
    || (echo "ERROR: HIP not linked - build failed" && exit 1)

# Stage artifacts for runtime image - include ROCm libs needed at runtime
RUN mkdir -p /staging/bin /staging/lib/rocm \
    && cp -r build/bin/* /staging/bin/ 2>/dev/null || true \
    && find build -name "*.so" -exec cp {} /staging/lib/ \; 2>/dev/null || true \
    && echo "Copying ROCm runtime libraries..." \
    && cp -aL /opt/rocm/lib/libhipblas.so* /staging/lib/rocm/ \
    && cp -aL /opt/rocm/lib/librocblas.so* /staging/lib/rocm/ \
    && cp -aL /opt/rocm/lib/libamdhip64.so* /staging/lib/rocm/ \
    && cp -aL /opt/rocm/lib/libhsa-runtime64.so* /staging/lib/rocm/ \
    && cp -aL /opt/rocm/lib/libamd_comgr.so* /staging/lib/rocm/ \
    && cp -aL /opt/rocm/lib/libhiprtc.so* /staging/lib/rocm/ 2>/dev/null || true \
    && cp -aL /opt/rocm/lib/librocsolver.so* /staging/lib/rocm/ 2>/dev/null || true \
    && cp -aL /opt/rocm/lib/librocsparse.so* /staging/lib/rocm/ 2>/dev/null || true \
    && cp -aL /opt/rocm/lib/librocprim.so* /staging/lib/rocm/ 2>/dev/null || true \
    && echo "Staged binaries:" && ls -la /staging/bin \
    && echo "Staged ROCm libs:" && ls -la /staging/lib/rocm/

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
# Copy ROCm libraries and create symlinks
COPY --from=builder /staging/lib/rocm/ /opt/rocm/lib/

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

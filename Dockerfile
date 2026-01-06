# PowerInfer with ROCm for AMD Strix Halo (gfx1151)
# Uses custom Strix Halo toolbox image with rocWMMA support
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest

ARG BUILD_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4-rocwmma
ARG RUNTIME_IMAGE=docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-6.4.4-rocwmma

FROM ${BUILD_IMAGE} AS builder

ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main
ARG AMDGPU_TARGETS=gfx1151

# Environment for ROCm build
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    GPU_TARGETS=${AMDGPU_TARGETS} \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    CMAKE_PREFIX_PATH=/opt/rocm \
    PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/bin:/bin:${PATH}"

WORKDIR /build

# Install build dependencies
# Base image is Fedora-based (ROCm uses Fedora/RHEL), so use dnf
# Include gcc/clang as fallback compilers for Python packages that need to compile C extensions
RUN dnf install -y \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    && dnf clean all

# Clone PowerInfer
RUN git clone --depth 1 --branch ${POWERINFER_BRANCH} ${POWERINFER_REPO} powerinfer

WORKDIR /build/powerinfer

# Install Python dependencies
# Unset CC/CXX to use system gcc for Python C extension builds (like cvxopt)
# ROCm clang may not be available or may not work for general C compilation
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Verify HIP/ROCm is available and find compilers
RUN echo "=== Checking ROCm/HIP installation ===" && \
    ls -la /opt/rocm/bin/ 2>/dev/null | head -20 || echo "No /opt/rocm/bin found" && \
    ls -la /opt/rocm/llvm/bin/ 2>/dev/null | head -20 || echo "No /opt/rocm/llvm/bin found" && \
    echo "=== Looking for compilers ===" && \
    (which hipcc && hipcc --version) || echo "hipcc not in PATH" && \
    (ls -la /opt/rocm/bin/hipcc 2>/dev/null) || echo "No /opt/rocm/bin/hipcc" && \
    (ls -la /opt/rocm/bin/amdclang* 2>/dev/null) || echo "No amdclang in /opt/rocm/bin" && \
    (ls -la /opt/rocm/llvm/bin/clang* 2>/dev/null | head -5) || echo "No clang in /opt/rocm/llvm/bin"

# Configure with CMake - Enable HIPBLAS for ROCm
# Use hipcc as C/CXX compiler to ensure AMD-specific flags like -munsafe-fp-atomics work
# hipcc is the HIP compiler wrapper that handles both C and C++ with proper ROCm flags
# Enable server build for HTTP API serving
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DLLAMA_HIPBLAS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS} \
    2>&1 | tee cmake_config.log \
    && echo "=== CMake Configuration Summary ===" \
    && grep -i "hipblas\|hip\|rocm\|gpu\|target\|arch\|compiler\|server" cmake_config.log || true

# Build PowerInfer
# Use pipefail to catch build failures even when piping to tee
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -e && \
    cmake --build build --config Release -j$(nproc) 2>&1 | tee build.log && \
    echo "=== Build completed, verifying binaries ===" && \
    ls -la build/bin/ && \
    echo "=== Available executables ===" && \
    find build/bin -type f -executable -ls && \
    if [ ! -f build/bin/main ]; then echo "ERROR: main binary not found!" && cat build.log && exit 1; fi && \
    echo "=== Checking HIP libraries ===" && \
    ldd build/bin/main 2>/dev/null | grep -i hip || echo "Note: HIP check inconclusive"

# Stage artifacts - copy all binaries and shared libraries
RUN set -e && \
    mkdir -p /staging/bin /staging/lib && \
    cp -v build/bin/* /staging/bin/ 2>/dev/null || true && \
    find build -name "*.so" -exec cp -v {} /staging/lib/ \; 2>/dev/null || true && \
    echo "=== Staged binaries ===" && ls -la /staging/bin && \
    echo "=== Staged libraries ===" && ls -la /staging/lib

# Runtime stage
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS runtime

ARG AMDGPU_TARGETS=gfx1151

# Runtime environment - matching eh-ops-repo working config
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    GPU_TARGETS=${AMDGPU_TARGETS} \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH="/opt/rocm/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/app"

# Copy binaries
COPY --from=builder /staging/bin/ /app/
COPY --from=builder /staging/lib/ /app/

WORKDIR /app

# Create models directory
RUN mkdir -p /models

# Default command
CMD ["./main", "--help"]

# Labels
LABEL maintainer="PowerInfer ROCm Build" \
      description="PowerInfer with ROCm 6.4.4 for AMD Strix Halo (gfx1151)" \
      rocm.version="6.4.4" \
      gpu.target="gfx1151"

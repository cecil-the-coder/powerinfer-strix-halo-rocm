# PowerInfer with ROCm for AMD Strix Halo (gfx1151) / Strix Point (gfx1150)
# Based on ROCm 7.1.1 optimized for Strix Halo/Point UMA architecture
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest
#
# NOTE: PowerInfer is based on an older llama.cpp that is NOT compatible with
# ROCm 7.x's updated hipBLAS API. The hipblasGemmEx function signature changed
# in ROCm 7.x to use hipblasComputeType_t instead of hipDataType.
# This Dockerfile applies a patch to fix the API compatibility.

# Use official ROCm dev image for building (has LLVM compiler + all dev tools)
ARG BUILD_IMAGE=rocm/dev-ubuntu-22.04:7.1.1
# Use same ROCm image for runtime to ensure library compatibility
ARG RUNTIME_IMAGE=rocm/dev-ubuntu-22.04:7.1.1

FROM ${BUILD_IMAGE} AS builder

# Build arguments
ARG AMDGPU_TARGETS=gfx1151
ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main

# Environment for ROCm build
# Set GPU targets via environment for HIP package detection
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    GPU_TARGETS=${AMDGPU_TARGETS} \
    HIP_VISIBLE_DEVICES=0 \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
    HIPCXX=/opt/rocm/llvm/bin/clang++ \
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
    patch \
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

# ============================================================================
# CRITICAL FIX: Patch PowerInfer's ggml-cuda.cu for ROCm 7.x API compatibility
# ============================================================================
# ROCm 7.x changed hipBLAS API: hipblasGemmEx now requires hipblasComputeType_t
# instead of hipDataType for the compute_type parameter.
#
# The old API (ROCm 6.x):
#   hipblasGemmEx(..., hipDataType computeType, ...)
# The new API (ROCm 7.x):
#   hipblasGemmEx(..., hipblasComputeType_t computeType, ...)
#
# We need to:
# 1. Add macro to convert CUDA compute types to hipblasComputeType_t
# 2. Update the GEMM function calls to use the correct type
# ============================================================================

# Create patch for ROCm 7.x hipBLAS API compatibility
RUN cat > /tmp/rocm7_hipblas_fix.patch << 'PATCH_EOF'
--- a/ggml-cuda.cu
+++ b/ggml-cuda.cu
@@ -28,6 +28,19 @@
 #define cudaDataType_t hipblasDatatype_t
 #define CUDA_R_16F HIPBLAS_R_16F
 #define CUDA_R_32F HIPBLAS_R_32F
+
+// ROCm 7.x API compatibility: hipblasGemmEx now uses hipblasComputeType_t
+// instead of hipDataType for the compute_type parameter
+#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)
+// ROCm 7.x with HIPBLAS_V2: use hipblasComputeType_t
+#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
+#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
+#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
+#else
+// ROCm 6.x or earlier: use hipDataType (mapped from CUDA types)
+#define CUBLAS_COMPUTE_16F CUDA_R_16F
+#define CUBLAS_COMPUTE_32F CUDA_R_32F
+#define CUBLAS_COMPUTE_32F_FAST_16F CUDA_R_32F
+#endif
 #define cublasGemmEx hipblasGemmEx
 #define cublasGemmBatchedEx hipblasGemmBatchedEx
 #define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
PATCH_EOF

# Apply the patch (allow fuzzy matching for line number differences)
RUN cd /build/powerinfer && \
    if patch -p1 --dry-run < /tmp/rocm7_hipblas_fix.patch 2>/dev/null; then \
        patch -p1 < /tmp/rocm7_hipblas_fix.patch; \
        echo "Patch applied successfully"; \
    else \
        echo "Patch failed to apply cleanly, attempting manual fix..."; \
        # Manual sed-based fix as fallback
        sed -i '/#define CUDA_R_32F HIPBLAS_R_32F/a \
\
// ROCm 7.x API compatibility: hipblasGemmEx now uses hipblasComputeType_t\
#if defined(__HIP_PLATFORM_AMD__)\
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F\
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F\
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F\
#endif' ggml-cuda.cu; \
        echo "Manual fix applied"; \
    fi && \
    echo "=== Verifying fix was applied ===" && \
    grep -A5 "CUDA_R_32F" ggml-cuda.cu | head -20

# Verify HIP is available before building
RUN hipcc --version && echo "HIP compiler found"

# Configure with CMake
# Note: PowerInfer's CMakeLists.txt uses LLAMA_HIPBLAS=ON to enable HIP support
# The GPU target is picked up from the environment via find_package(hip)
# We also pass -DHIPBLAS_V2=1 to enable the new ROCm 7.x API
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DLLAMA_HIPBLAS=ON \
    -DCMAKE_CXX_FLAGS="-DHIPBLAS_V2=1" \
    -DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS} \
    2>&1 | tee cmake_config.log \
    && echo "=== CMake Configuration Summary ===" \
    && grep -i "hipblas\|hip\|rocm\|gpu\|target\|arch" cmake_config.log || true

# Build and verify HIP was actually used
RUN cmake --build build --config Release -j$(nproc) 2>&1 | tee build.log \
    && echo "=== Checking if HIP libraries are linked ===" \
    && (ldd build/bin/main 2>/dev/null | grep -i hip && echo "HIP libraries linked successfully") \
    || (echo "WARNING: Could not verify HIP linking via ldd - checking build output..." \
        && grep -i "ggml-rocm\|hipblas\|rocblas" build.log \
        && echo "Build appears to include HIP components")

# Stage artifacts for runtime image - include ROCm libs needed at runtime
RUN mkdir -p /staging/bin /staging/lib/rocm /staging/rocblas \
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
    && echo "Copying rocBLAS Tensile library (GPU kernels)..." \
    && cp -r /opt/rocm/lib/rocblas /staging/rocblas/ \
    && echo "Staged binaries:" && ls -la /staging/bin \
    && echo "Staged ROCm libs:" && ls -la /staging/lib/rocm/ \
    && echo "Staged rocBLAS library:" && ls -la /staging/rocblas/

# Runtime stage - smaller image using same ROCm image
ARG RUNTIME_IMAGE
ARG AMDGPU_TARGETS=gfx1151
FROM ${RUNTIME_IMAGE} AS runtime

# Re-declare ARG after FROM to use in this stage
ARG AMDGPU_TARGETS

# Runtime environment
# HSA_OVERRIDE_GFX_VERSION helps with GPU detection for newer architectures
# For gfx1150: 11.5.0, for gfx1151: 11.5.1
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    AMDGPU_TARGETS=${AMDGPU_TARGETS} \
    GPU_TARGETS=${AMDGPU_TARGETS} \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH="/opt/rocm/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/app"

# Copy staged binaries and libraries
COPY --from=builder /staging/bin/ /app/
COPY --from=builder /staging/lib/ /app/
# Copy ROCm libraries and create symlinks
COPY --from=builder /staging/lib/rocm/ /opt/rocm/lib/
# Copy rocBLAS Tensile library (required for GPU kernels)
COPY --from=builder /staging/rocblas/rocblas/ /opt/rocm/lib/rocblas/

WORKDIR /app

# Create models directory
RUN mkdir -p /models

# Default command - show help
CMD ["./main", "--help"]

# Labels
LABEL maintainer="PowerInfer ROCm Build" \
      description="PowerInfer with ROCm 7.1.1 for AMD Strix Point/Halo (gfx1150/gfx1151)" \
      rocm.version="7.1.1" \
      gpu.target="${AMDGPU_TARGETS}"

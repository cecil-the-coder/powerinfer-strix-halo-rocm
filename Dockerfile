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
# ROCm 7.x (hipBLAS 3.0+) changed the hipBLAS API:
# - hipblasGemmEx, hipblasGemmBatchedEx, hipblasGemmStridedBatchedEx now require
#   hipblasComputeType_t instead of hipDataType for the compute_type parameter.
#
# The old API (ROCm 6.x / hipBLAS < 3.0):
#   hipblasGemmEx(..., hipDataType computeType, ...)
# The new API (ROCm 7.x / hipBLAS 3.0+):
#   hipblasGemmEx(..., hipblasComputeType_t computeType, ...)
#
# PowerInfer uses CUDA_R_16F/CUDA_R_32F (mapped to hipDataType) as the compute
# type in GEMM calls. We need to create wrapper functions that:
# 1. Convert hipDataType compute_type to hipblasComputeType_t
# 2. Call the underlying hipBLAS function with the correct type
# ============================================================================

# Create comprehensive fix for ROCm 7.x hipBLAS API compatibility
# This creates inline wrapper functions that handle the type conversion
# IMPORTANT: We use inline functions with _rocm7_compat suffix to avoid recursion
RUN cat > /tmp/rocm7_hipblas_fix.h << 'WRAPPER_EOF'
// ROCm 7.x API compatibility wrappers for hipBLAS GEMM functions
// hipBLAS 3.0+ changed compute_type parameter from hipDataType to hipblasComputeType_t
//
// This header MUST be included BEFORE any hipBLAS macros are defined, so that
// we can call the real hipblasGemmEx functions inside our wrappers.

#ifndef ROCM7_HIPBLAS_FIX_H
#define ROCM7_HIPBLAS_FIX_H

#if defined(__HIP_PLATFORM_AMD__) && defined(HIPBLAS_V2)

#include <hipblas/hipblas.h>

// Helper to convert hipDataType to hipblasComputeType_t
static inline hipblasComputeType_t ggml_hipDataTypeToComputeType(hipDataType dtype) {
    switch (dtype) {
        case HIP_R_16F:
            return HIPBLAS_COMPUTE_16F;
        case HIP_R_32F:
            return HIPBLAS_COMPUTE_32F;
        case HIP_R_64F:
            return HIPBLAS_COMPUTE_64F;
        default:
            return HIPBLAS_COMPUTE_32F;  // Safe default
    }
}

// Wrapper for hipblasGemmEx that accepts hipDataType for compute_type
// and converts it to hipblasComputeType_t
// Named with _rocm7_compat suffix to avoid macro expansion issues
static inline hipblasStatus_t ggml_hipblasGemmEx_rocm7_compat(
    hipblasHandle_t handle,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    int m, int n, int k,
    const void* alpha,
    const void* A, hipDataType aType, int lda,
    const void* B, hipDataType bType, int ldb,
    const void* beta,
    void* C, hipDataType cType, int ldc,
    hipDataType computeType,  // Accept hipDataType for compatibility
    hipblasGemmAlgo_t algo)
{
    // Call the real hipblasGemmEx with converted compute type
    return ::hipblasGemmEx(
        handle, transA, transB,
        m, n, k,
        alpha,
        A, aType, lda,
        B, bType, ldb,
        beta,
        C, cType, ldc,
        ggml_hipDataTypeToComputeType(computeType),  // Convert to hipblasComputeType_t
        algo);
}

// Wrapper for hipblasGemmBatchedEx
static inline hipblasStatus_t ggml_hipblasGemmBatchedEx_rocm7_compat(
    hipblasHandle_t handle,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    int m, int n, int k,
    const void* alpha,
    const void* const A[], hipDataType aType, int lda,
    const void* const B[], hipDataType bType, int ldb,
    const void* beta,
    void* const C[], hipDataType cType, int ldc,
    int batchCount,
    hipDataType computeType,  // Accept hipDataType for compatibility
    hipblasGemmAlgo_t algo)
{
    // Cast to match hipBLAS expected types (const void** and void**)
    return ::hipblasGemmBatchedEx(
        handle, transA, transB,
        m, n, k,
        alpha,
        (const void**)A, aType, lda,
        (const void**)B, bType, ldb,
        beta,
        (void**)C, cType, ldc,
        batchCount,
        ggml_hipDataTypeToComputeType(computeType),  // Convert to hipblasComputeType_t
        algo);
}

// Wrapper for hipblasGemmStridedBatchedEx
static inline hipblasStatus_t ggml_hipblasGemmStridedBatchedEx_rocm7_compat(
    hipblasHandle_t handle,
    hipblasOperation_t transA,
    hipblasOperation_t transB,
    int m, int n, int k,
    const void* alpha,
    const void* A, hipDataType aType, int lda, hipblasStride strideA,
    const void* B, hipDataType bType, int ldb, hipblasStride strideB,
    const void* beta,
    void* C, hipDataType cType, int ldc, hipblasStride strideC,
    int batchCount,
    hipDataType computeType,  // Accept hipDataType for compatibility
    hipblasGemmAlgo_t algo)
{
    return ::hipblasGemmStridedBatchedEx(
        handle, transA, transB,
        m, n, k,
        alpha,
        A, aType, lda, strideA,
        B, bType, ldb, strideB,
        beta,
        C, cType, ldc, strideC,
        batchCount,
        ggml_hipDataTypeToComputeType(computeType),  // Convert to hipblasComputeType_t
        algo);
}

#endif // __HIP_PLATFORM_AMD__ && HIPBLAS_V2

#endif // ROCM7_HIPBLAS_FIX_H
WRAPPER_EOF

# Apply the fix:
# 1. Copy the wrapper header to the source directory
# 2. Create a Python script to apply the patches (more reliable than sed for multi-line)
# 3. Modify the macro definitions to use our wrappers when HIPBLAS_V2 is defined

# First, create the Python patch script
RUN cat > /tmp/apply_rocm7_patch.py << 'PYEOF'
import sys

with open('ggml-cuda.cu', 'r') as f:
    content = f.read()

# Check if already patched
if 'rocm7_hipblas_fix.h' in content:
    print("ROCm 7.x compatibility fix already present, skipping")
    sys.exit(0)

print("Applying ROCm 7.x hipBLAS API compatibility fix...")

# 1. Add include for rocm7_hipblas_fix.h after #if defined(GGML_USE_HIPBLAS)
include_patch = '''#if defined(GGML_USE_HIPBLAS)
// ROCm 7.x hipBLAS API compatibility - must include before macro definitions
#if defined(HIPBLAS_V2)
#include "rocm7_hipblas_fix.h"
#endif'''
content = content.replace('#if defined(GGML_USE_HIPBLAS)', include_patch, 1)

# 2. Replace cublasGemmEx macro with conditional version
old_gemm = '#define cublasGemmEx hipblasGemmEx'
new_gemm = '''#if defined(HIPBLAS_V2)
#define cublasGemmEx ggml_hipblasGemmEx_rocm7_compat
#else
#define cublasGemmEx hipblasGemmEx
#endif'''
content = content.replace(old_gemm, new_gemm, 1)

# 3. Replace cublasGemmBatchedEx macro with conditional version
old_batched = '#define cublasGemmBatchedEx hipblasGemmBatchedEx'
new_batched = '''#if defined(HIPBLAS_V2)
#define cublasGemmBatchedEx ggml_hipblasGemmBatchedEx_rocm7_compat
#else
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#endif'''
content = content.replace(old_batched, new_batched, 1)

# 4. Replace cublasGemmStridedBatchedEx macro with conditional version
old_strided = '#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx'
new_strided = '''#if defined(HIPBLAS_V2)
#define cublasGemmStridedBatchedEx ggml_hipblasGemmStridedBatchedEx_rocm7_compat
#else
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#endif'''
content = content.replace(old_strided, new_strided, 1)

with open('ggml-cuda.cu', 'w') as f:
    f.write(content)

print("Patches applied successfully")
PYEOF

# Now apply the patch
RUN cd /build/powerinfer && \
    cp /tmp/rocm7_hipblas_fix.h /build/powerinfer/rocm7_hipblas_fix.h && \
    python3 /tmp/apply_rocm7_patch.py && \
    echo "=== Verifying fix was applied ===" && \
    grep -n "rocm7_hipblas_fix\|HIPBLAS_V2\|ggml_hipblas" ggml-cuda.cu | head -30

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

# Build PowerInfer - MUST succeed
# Use set -e to ensure any failure stops the build
RUN set -e && \
    cmake --build build --config Release -j$(nproc) 2>&1 | tee build.log && \
    echo "=== Build completed, verifying binaries ===" && \
    ls -la build/bin/ && \
    if [ ! -f build/bin/main ]; then echo "ERROR: Build failed - main binary not found!" && cat build.log && false; fi && \
    echo "=== Checking if HIP libraries are linked ===" && \
    (ldd build/bin/main 2>/dev/null | grep -i hip && echo "HIP libraries linked successfully" || echo "Note: HIP linking check inconclusive")

# Stage artifacts for runtime image - include ROCm libs needed at runtime
# CRITICAL: Fail build if binaries don't exist
RUN set -e && \
    mkdir -p /staging/bin /staging/lib/rocm /staging/rocblas && \
    echo "=== Checking for built binaries ===" && \
    ls -la build/bin/ && \
    if [ ! -f build/bin/main ]; then echo "ERROR: build/bin/main not found - build failed!" && false; fi && \
    cp -r build/bin/* /staging/bin/ && \
    find build -name "*.so" -exec cp {} /staging/lib/ \; 2>/dev/null; \
    echo "Copying ROCm runtime libraries..." && \
    cp -aL /opt/rocm/lib/libhipblas.so* /staging/lib/rocm/ && \
    cp -aL /opt/rocm/lib/librocblas.so* /staging/lib/rocm/ && \
    cp -aL /opt/rocm/lib/libamdhip64.so* /staging/lib/rocm/ && \
    cp -aL /opt/rocm/lib/libhsa-runtime64.so* /staging/lib/rocm/ && \
    cp -aL /opt/rocm/lib/libamd_comgr.so* /staging/lib/rocm/ && \
    (cp -aL /opt/rocm/lib/libhiprtc.so* /staging/lib/rocm/ 2>/dev/null || true) && \
    (cp -aL /opt/rocm/lib/librocsolver.so* /staging/lib/rocm/ 2>/dev/null || true) && \
    (cp -aL /opt/rocm/lib/librocsparse.so* /staging/lib/rocm/ 2>/dev/null || true) && \
    (cp -aL /opt/rocm/lib/librocprim.so* /staging/lib/rocm/ 2>/dev/null || true) && \
    echo "Copying rocBLAS Tensile library (GPU kernels)..." && \
    cp -r /opt/rocm/lib/rocblas /staging/rocblas/ && \
    echo "Staged binaries:" && ls -la /staging/bin && \
    echo "Staged ROCm libs:" && ls -la /staging/lib/rocm/ && \
    echo "Staged rocBLAS library:" && ls -la /staging/rocblas/

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

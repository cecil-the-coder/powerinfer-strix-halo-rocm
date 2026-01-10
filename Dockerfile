# PowerInfer with ROCm 6.4.4 for AMD Strix Halo (gfx1151)
# Based on kyuz0/amd-strix-halo-toolboxes approach
# Note: ROCm 7.x has breaking hipBLAS API changes incompatible with PowerInfer
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest

# Build stage - Fedora with ROCm from repo
FROM registry.fedoraproject.org/fedora:43 AS builder

# ROCm 6.4.4 repo (newest version compatible with PowerInfer's hipBLAS API)
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-6.4.4]
name=ROCm6.4.4
baseurl=https://repo.radeon.com/rocm/el9/6.4.4/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

# Build dependencies - including ROCm compilers and HIP
RUN dnf -y --nodocs --setopt=install_weak_deps=False \
    --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
    install \
    make gcc cmake lld clang clang-devel compiler-rt libcurl-devel ninja-build \
    rocm-llvm rocm-device-libs hip-runtime-amd hip-devel \
    rocblas rocblas-devel hipblas hipblas-devel rocm-cmake libomp-devel libomp \
    rocminfo \
    git-core python3 python3-pip python3-devel \
    && dnf clean all && rm -rf /var/cache/dnf/*

# ROCm environment
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

# Build rocWMMA for gfx1151 (native Strix Halo support)
WORKDIR /opt
COPY build-rocwmma.sh .
RUN chmod +x build-rocwmma.sh && ./build-rocwmma.sh

# Clone PowerInfer
ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main

WORKDIR /opt/powerinfer
RUN git clone --depth 1 --branch ${POWERINFER_BRANCH} ${POWERINFER_REPO} .

# Apply gfx1151 (Strix Halo) compatibility patches
# These fix warp mask type issues that cause memory access faults on gfx1151
COPY patches/ /opt/patches/
RUN chmod +x /opt/patches/apply-gfx1151-fix.sh && \
    /opt/patches/apply-gfx1151-fix.sh /opt/powerinfer

# Apply null pointer crash fixes for sparse inference on gfx1151
# These fix segfaults during KV cache initialization and sparse tensor handling
RUN python3 /opt/patches/apply-null-pointer-fixes.py /opt/powerinfer

# Install Python dependencies (optional)
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Install powerinfer Python module for GPU split generation
RUN pip3 install --no-cache-dir torch numpy cvxopt gguf && \
    cd /opt/powerinfer/powerinfer-py && pip3 install -e .

# Build PowerInfer with HIP support for gfx1151
# Must explicitly set CC/CXX to ROCm compilers to handle HIP-specific flags
# Include hip_shfl_fix.h for proper warp shuffle compatibility
RUN CC=/opt/rocm/llvm/bin/amdclang \
    CXX=/opt/rocm/llvm/bin/amdclang++ \
    cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/amdclang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
    -DLLAMA_HIPBLAS=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DLLAMA_HIP_UMA=ON \
    -DROCM_PATH=/opt/rocm \
    -DHIP_PATH=/opt/rocm \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -include /opt/patches/hip_shfl_fix.h" \
    && cmake --build build --config Release -- -j$(nproc)

# Verify build
RUN echo "=== Built binaries ===" && \
    ls -la build/bin/ && \
    if [ ! -f build/bin/main ]; then echo "ERROR: main not found" && exit 1; fi

# Copy libs
RUN find /opt/powerinfer/build -type f -name 'lib*.so*' -exec cp {} /usr/lib64/ \; && ldconfig


# Runtime stage - minimal Fedora with ROCm runtime
FROM registry.fedoraproject.org/fedora-minimal:43

# ROCm 6.4.4 repo
RUN <<'EOF'
tee /etc/yum.repos.d/rocm.repo <<REPO
[ROCm-6.4.4]
name=ROCm6.4.4
baseurl=https://repo.radeon.com/rocm/el9/6.4.4/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
EOF

# Runtime dependencies only (including Python for GPU split generation)
RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
    --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
    install \
    bash ca-certificates libatomic libstdc++ libgcc libgomp \
    hip-runtime-amd rocblas hipblas \
    rocminfo \
    python3 python3-pip \
    && microdnf clean all && rm -rf /var/cache/dnf/*

# Install Python packages for GPU split generation
# cvxopt needs gcc to compile, so install it temporarily
RUN microdnf -y install gcc python3-devel && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir numpy cvxopt gguf && \
    microdnf -y remove gcc python3-devel && \
    microdnf clean all && rm -rf /var/cache/dnf/*

# Copy binaries and libraries from builder
COPY --from=builder /opt/powerinfer/build/bin/ /app/
COPY --from=builder /usr/lib64/libllama*.so* /usr/lib64/
COPY --from=builder /usr/lib64/libggml*.so* /usr/lib64/

# Copy powerinfer Python module
COPY --from=builder /opt/powerinfer/powerinfer-py /opt/powerinfer-py
RUN pip3 install -e /opt/powerinfer-py

# Library paths
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
    && echo "/app" >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

# Runtime environment
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH="/opt/rocm/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/app:/usr/lib64"

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

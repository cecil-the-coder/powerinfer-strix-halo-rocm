# SmallThinker MoE inference for AMD Strix Halo (gfx1151)
# ROCm 7 + GGML HIP backend (no legacy hipBLAS API dependency)
#
# /app/server — SmallThinker MoE llama-server (powerinfer-fused-sparse-moe)
#
# Build: docker build -t powerinfer-rocm:latest .
# Run:   docker run --device=/dev/kfd --device=/dev/dri -v /models:/models powerinfer-rocm:latest

# Build stage
FROM registry.fedoraproject.org/fedora:43 AS builder

# ROCm 7 repo
RUN tee /etc/yum.repos.d/rocm.repo <<'REPO'
[ROCm]
name=ROCm7
baseurl=https://repo.radeon.com/rocm/el9/7.0.0/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO

RUN dnf -y --nodocs --setopt=install_weak_deps=False \
    --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
    install \
    make gcc cmake lld clang clang-devel compiler-rt libcurl-devel ninja-build \
    rocm-llvm rocm-device-libs hip-runtime-amd hip-devel \
    rocblas rocblas-devel hipblas hipblas-devel rocm-cmake libomp-devel libomp \
    git-core \
    libaio-devel \
    && dnf clean all && rm -rf /var/cache/dnf/*

# Build liburing from source (headers + static lib)
# Fedora 43 liburing-devel omits liburing.a; the smallthinker build requires -Wl,-Bstatic -luring
RUN git clone --depth 1 https://github.com/axboe/liburing.git /tmp/liburing-src \
    && cd /tmp/liburing-src \
    && ./configure --prefix=/usr --libdir=/usr/lib64 --includedir=/usr/include \
    && make install \
    && rm -rf /tmp/liburing-src

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

# Clone PowerInfer (smallthinker is a submodule)
ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main

WORKDIR /opt/powerinfer
RUN git clone --depth 1 --branch ${POWERINFER_BRANCH} ${POWERINFER_REPO} .

COPY patches/ /opt/patches/

# Build SmallThinker MoE server
WORKDIR /opt/powerinfer/smallthinker
RUN git submodule update --init --recursive && \
    cp /opt/patches/powerinfer-cuda-stub.h ggml/src/ggml-cuda/powerinfer-cuda.h

RUN export LIBRARY_PATH="/usr/lib64:${LIBRARY_PATH}" && \
    CC=/opt/rocm/llvm/bin/amdclang \
    CXX=/opt/rocm/llvm/bin/amdclang++ \
    cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/amdclang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_HIP_UMA=ON \
    -DROCM_PATH=/opt/rocm \
    -DHIP_PATH=/opt/rocm \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -include /opt/patches/hip_shfl_fix.h" \
    -DCMAKE_SKIP_INSTALL_RULES=TRUE \
    && cmake --build build --config Release --target llama-server -- -j$(nproc)

RUN if [ ! -f build/bin/llama-server ]; then echo "ERROR: llama-server not found" && exit 1; fi

# Collect shared libs from the SmallThinker build
RUN find /opt/powerinfer/smallthinker/build -type f -name 'lib*.so*' -exec cp {} /usr/lib64/ \; && \
    ldconfig


# Runtime stage - minimal Fedora with ROCm runtime
FROM registry.fedoraproject.org/fedora-minimal:43

# ROCm 7 repo
RUN tee /etc/yum.repos.d/rocm.repo <<'REPO'
[ROCm]
name=ROCm7
baseurl=https://repo.radeon.com/rocm/el9/7.0.0/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO

RUN microdnf -y --nodocs --setopt=install_weak_deps=0 \
    --exclude='*sdk*' --exclude='*samples*' --exclude='*-doc*' --exclude='*-docs*' \
    install \
    bash ca-certificates libatomic libstdc++ libgcc libgomp libaio \
    hip-runtime-amd rocblas hipblas \
    && microdnf clean all && rm -rf /var/cache/dnf/*

COPY --from=builder /opt/powerinfer/smallthinker/build/bin/llama-server /app/server
COPY --from=builder /usr/lib64/libllama*.so* /usr/lib64/
COPY --from=builder /usr/lib64/libggml*.so* /usr/lib64/
COPY --from=builder /usr/lib64/libmtmd*.so* /usr/lib64/
COPY --from=builder /usr/lib64/libpowerinfer*.so* /usr/lib64/

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && echo "/usr/local/lib64" >> /etc/ld.so.conf.d/local.conf \
    && echo "/app" >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    LLAMA_HIP_UMA=ON \
    ROCBLAS_USE_HIPBLASLT=1 \
    PATH="/opt/rocm/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/app:/usr/lib64"

WORKDIR /app
RUN mkdir -p /models

CMD ["/app/server", "--help"]

LABEL maintainer="PowerInfer ROCm Build" \
      description="SmallThinker MoE inference with ROCm 7 for AMD Strix Halo (gfx1151)" \
      rocm.version="7.0.0" \
      gpu.target="gfx1151"

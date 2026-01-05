# PowerInfer CPU-only build
# ROCm/HIP is disabled due to memory faults on AMD Strix Halo (gfx1151)
# See: https://github.com/ROCm/ROCm/issues/5824
#
# Build: docker build -t powerinfer:latest .
# Run:   docker run -v /models:/models powerinfer:latest
#
# TODO: Re-enable HIPBLAS when ROCm properly supports gfx1151

ARG BUILD_IMAGE=ubuntu:22.04
ARG RUNTIME_IMAGE=ubuntu:22.04

FROM ${BUILD_IMAGE} AS builder

ARG POWERINFER_REPO=https://github.com/SJTU-IPADS/PowerInfer.git
ARG POWERINFER_BRANCH=main

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone PowerInfer
RUN git clone --depth 1 --branch ${POWERINFER_BRANCH} ${POWERINFER_REPO} powerinfer

WORKDIR /build/powerinfer

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Configure with CMake - CPU only with OpenBLAS
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BLAS=ON \
    -DLLAMA_BLAS_VENDOR=OpenBLAS \
    2>&1 | tee cmake_config.log

# Build PowerInfer
RUN set -e && \
    cmake --build build --config Release -j$(nproc) 2>&1 | tee build.log && \
    echo "=== Build completed, verifying binaries ===" && \
    ls -la build/bin/ && \
    if [ ! -f build/bin/main ]; then echo "ERROR: Build failed!" && cat build.log && false; fi

# Stage artifacts for runtime image
RUN set -e && \
    mkdir -p /staging/bin /staging/lib && \
    cp -r build/bin/* /staging/bin/ && \
    find build -name "*.so" -exec cp {} /staging/lib/ \; 2>/dev/null || true && \
    echo "Staged binaries:" && ls -la /staging/bin

# Runtime stage - minimal image
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries
COPY --from=builder /staging/bin/ /app/
COPY --from=builder /staging/lib/ /app/

WORKDIR /app
ENV PATH="/app:${PATH}"

# Create models directory
RUN mkdir -p /models

# Default command - show help
CMD ["./main", "--help"]

# Labels
LABEL maintainer="PowerInfer Build" \
      description="PowerInfer CPU-only (ROCm disabled due to gfx1151 issues)"

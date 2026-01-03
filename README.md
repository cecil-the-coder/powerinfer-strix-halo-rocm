# PowerInfer ROCm for AMD Strix

PowerInfer with ROCm/HIP support for AMD Strix APUs.

## Supported Hardware

| APU | Architecture | GPU Target | CUs |
|-----|--------------|------------|-----|
| Strix Halo (Ryzen AI Max) | RDNA 3.5 | gfx1151 | 40 |
| Strix Point | RDNA 3.5 | gfx1150 | 16 |

## Quick Start

### Pull Pre-built Image

```bash
# Strix Halo (gfx1151)
docker pull ghcr.io/cecil-the-coder/powerinfer-strix-halo-rocm:latest

# Strix Point (gfx1150)
docker pull ghcr.io/cecil-the-coder/powerinfer-strix-halo-rocm:gfx1150-latest
```

### Run Inference

```bash
docker run --device=/dev/kfd --device=/dev/dri \
    -v /path/to/models:/models \
    ghcr.io/cecil-the-coder/powerinfer-strix-halo-rocm:latest \
    ./main -m /models/your-model.gguf \
    -p "Hello, world!" \
    -n 128
```

### Run Server

```bash
docker run --device=/dev/kfd --device=/dev/dri \
    -v /path/to/models:/models \
    -p 8080:8080 \
    ghcr.io/cecil-the-coder/powerinfer-strix-halo-rocm:latest \
    ./server -m /models/your-model.gguf \
    --host 0.0.0.0 --port 8080
```

## Deploy with Docker Compose

```bash
# Download a model first
mkdir -p models
huggingface-cli download Tiiny/ReluLLaMA-7B-PowerInfer-GGUF \
    --local-dir ./models

# Start the server
docker compose up
```

Edit `docker-compose.yaml` to change the model path or use a different GPU target.

## Deploy on Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
```

Edit `kubernetes/deployment.yaml` to configure the model path and GPU target.

## Build from Source

```bash
git clone https://github.com/cecil-the-coder/powerinfer-strix-halo-rocm.git
cd powerinfer-strix-halo-rocm

# Build for Strix Halo (default)
docker build -t powerinfer-rocm:latest .

# Build for Strix Point
docker build --build-arg AMDGPU_TARGETS=gfx1150 -t powerinfer-rocm:gfx1150 .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | GPU version override for ROCm |
| `ROCBLAS_USE_HIPBLASLT` | `1` | Use hipBLASLt for better performance |

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--vram-budget N` | GB of GPU memory for hot neurons |
| `-t N` | CPU threads for cold neuron computation |
| `-ngl N` | GPU layers (999 = all) |
| `-c N` | Context size |

## What is PowerInfer?

PowerInfer exploits activation sparsity in LLMs. During inference, ~70-90% of neurons are inactive. PowerInfer:

1. Precomputes which neurons are "hot" (frequently active) vs "cold" (rarely active)
2. Keeps hot neurons on GPU, cold neurons on CPU
3. Skips calculations for inactive neurons

**Note:** PowerInfer requires models with ReLU/ReGLU activation and precomputed activation statistics. Standard models using SiLU/SwiGLU won't benefit from sparsity optimizations but will still run.

## Compatible Models

| Model | Size | HuggingFace |
|-------|------|-------------|
| ReluLLaMA-7B | ~7GB | `Tiiny/ReluLLaMA-7B-PowerInfer-GGUF` |
| ReluLLaMA-13B | ~13GB | `Tiiny/ReluLLaMA-13B-PowerInfer-GGUF` |
| ReluLLaMA-70B | ~40GB | `Tiiny/ReluLLaMA-70B-PowerInfer-GGUF` |

## References

- [PowerInfer GitHub](https://github.com/SJTU-IPADS/PowerInfer)
- [PowerInfer Paper](https://arxiv.org/abs/2312.12456)
- [ROCm Documentation](https://rocm.docs.amd.com/)

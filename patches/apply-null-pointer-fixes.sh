#!/bin/bash
# Apply null pointer crash fixes for PowerInfer on gfx1151 (AMD Strix Halo)
# These fixes address segfaults during sparse inference initialization

set -euo pipefail

POWERINFER_DIR="${1:-/opt/powerinfer}"

echo "=== Applying null pointer crash fixes to PowerInfer ==="
echo "Directory: $POWERINFER_DIR"

# Verify PowerInfer directory exists
if [[ ! -d "$POWERINFER_DIR" ]]; then
    echo "Error: PowerInfer directory not found: $POWERINFER_DIR"
    exit 1
fi

# Check for required files
LLAMA_CPP="$POWERINFER_DIR/llama.cpp"
GGML_CUDA="$POWERINFER_DIR/ggml-cuda.cu"

if [[ ! -f "$LLAMA_CPP" ]]; then
    echo "Error: llama.cpp not found"
    exit 1
fi

if [[ ! -f "$GGML_CUDA" ]]; then
    echo "Error: ggml-cuda.cu not found"
    exit 1
fi

echo ""
echo "=== Fix 1: NULL check in add_tensor lambda (llama.cpp) ==="

# Backup
cp "$LLAMA_CPP" "$LLAMA_CPP.backup.nullfix"

# Fix the add_tensor lambda to check for NULL before dereferencing
if grep -q "if (t->backend == GGML_BACKEND_GPU" "$LLAMA_CPP"; then
    sed -i 's/if (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT)/if (t != nullptr \&\& (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT))/' "$LLAMA_CPP"
    echo "  Applied: NULL check in add_tensor lambda"
else
    echo "  Skipped: Pattern not found or already patched"
fi

echo ""
echo "=== Fix 2: NULL check for KV cache tensors (llama.cpp) ==="

# Add NULL check after ggml_new_tensor_1d for cache.k and cache.v
# We insert a check after the tensor creation
if grep -q 'cache.k = ggml_new_tensor_1d.*cache.ctx.*wtype.*n_elements' "$LLAMA_CPP" && \
   ! grep -q 'cache.k == nullptr || cache.v == nullptr' "$LLAMA_CPP"; then
    # Use a heredoc with sed to add the NULL check
    sed -i '/cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);/a\
\
    // FIX: Check for allocation failure before using tensors\
    if (cache.k == nullptr || cache.v == nullptr) {\
        LLAMA_LOG_ERROR("%s: failed to allocate KV cache tensors\\n", __func__);\
        ggml_free(cache.ctx);\
        return false;\
    }' "$LLAMA_CPP"
    echo "  Applied: NULL check for KV cache tensor allocation"
else
    echo "  Skipped: Already patched or pattern not found"
fi

echo ""
echo "=== Fix 3: Remove dangerous HIP self-pointer hack (llama.cpp) ==="

# Replace the dangerous self-pointer hack with proper NULL handling
# The hack sets up_gpu->data = up_gpu which causes memory corruption

# First occurrence in llm_build_sparse_mul_mat
if grep -q "up_gpu->data = up_gpu;" "$LLAMA_CPP"; then
    # Replace the entire HIPBLAS hack block with proper NULL check
    sed -i '/#ifdef GGML_USE_HIPBLAS/{
        N
        N
        N
        N
        N
        N
        N
        s/#ifdef GGML_USE_HIPBLAS\n\/\/ WARNING: THIS IS A HACK!.*\n.*\n.*\n.*\n    up_gpu->data = up_gpu;\n\n#endif/#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_CUBLAS)\n    \/\/ FIX: Properly handle NULL up_gpu for gfx1151 ROCm targets\n    \/\/ Instead of dangerous self-pointer hack, fall through to CPU path\n    if (up_gpu == nullptr) {\n        out = ggml_mul_mat_idx(ctx, up, inp, idx, gpu_index);\n        cb(out, full_name.c_str());\n        return out;\n    }\n#endif/
    }' "$LLAMA_CPP"
    echo "  Applied: Fixed first HIP self-pointer hack (up_gpu)"
else
    echo "  Skipped: First HIP hack not found or already patched"
fi

# Second occurrence in llm_build_sparse_axpy
if grep -q "wt_gpu->data = wt_gpu;" "$LLAMA_CPP"; then
    sed -i '/#ifdef GGML_USE_HIPBLAS/{
        N
        N
        N
        N
        N
        N
        N
        s/#ifdef GGML_USE_HIPBLAS\n\/\/ WARNING: THIS IS A HACK!.*\n.*\n.*\n.*\n    wt_gpu->data = wt_gpu;\n\n#endif/#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_CUBLAS)\n    \/\/ FIX: Properly handle NULL wt_gpu for gfx1151 ROCm targets\n    \/\/ Instead of dangerous self-pointer hack, fall through to CPU path\n    if (wt_gpu == nullptr) {\n        out = ggml_axpy(ctx, w_t, x, sparse_idx, gpu_index);\n        cb(out, full_name.c_str());\n        return out;\n    }\n#endif/
    }' "$LLAMA_CPP"
    echo "  Applied: Fixed second HIP self-pointer hack (wt_gpu)"
else
    echo "  Skipped: Second HIP hack not found or already patched"
fi

echo ""
echo "=== Fix 4: NULL check in ggml_cuda_assign_buffers_no_scratch ==="

cp "$GGML_CUDA" "$GGML_CUDA.backup.nullfix"

# Add NULL check to ggml_cuda_assign_buffers_no_scratch like ggml_cuda_assign_buffers has
if grep -q "void ggml_cuda_assign_buffers_no_scratch.*{" "$GGML_CUDA" && \
   ! grep -q "ggml_cuda_assign_buffers_no_scratch.*tensor.*NULL" "$GGML_CUDA"; then
    sed -i '/void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor \* tensor) {/a\
    // FIX: Add NULL check like ggml_cuda_assign_buffers() has\
    if (tensor == NULL)\
        return;' "$GGML_CUDA"
    echo "  Applied: NULL check in ggml_cuda_assign_buffers_no_scratch"
else
    echo "  Skipped: Already patched or pattern not found"
fi

echo ""
echo "=== Fix 5: Set g_device_count to 0 on failure ==="

# When cudaGetDeviceCount fails, set g_device_count to 0 instead of leaving it at -1
if grep -q "g_cublas_loaded = false;" "$GGML_CUDA" && \
   ! grep -q "g_device_count = 0;" "$GGML_CUDA"; then
    sed -i 's/initialized = true;\n.*g_cublas_loaded = false;/g_device_count = 0;\n            initialized = true;\n            g_cublas_loaded = false;/' "$GGML_CUDA"
    echo "  Applied: Set g_device_count = 0 on failure"
else
    echo "  Skipped: Already patched or pattern not found"
fi

echo ""
echo "=== All null pointer fixes applied ==="
echo ""
echo "Summary of changes:"
echo "  1. add_tensor lambda now checks for NULL before accessing t->backend"
echo "  2. KV cache tensor allocation now validates success"
echo "  3. Dangerous HIP self-pointer hacks replaced with CPU fallback"
echo "  4. ggml_cuda_assign_buffers_no_scratch now has NULL check"
echo "  5. g_device_count set to 0 when device enumeration fails"
echo ""
echo "Rebuild with: cmake --build build --config Release"

#!/usr/bin/env python3
"""
Apply null pointer crash fixes for PowerInfer on gfx1151 (AMD Strix Halo)
These fixes address segfaults during sparse inference initialization
"""

import os
import re
import sys
import shutil

def apply_fix(filepath, pattern, replacement, description, backup=True):
    """Apply a regex fix to a file"""
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if re.search(pattern, content, re.DOTALL):
        if backup and not os.path.exists(filepath + '.backup.nullfix'):
            shutil.copy2(filepath, filepath + '.backup.nullfix')

        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"  Applied: {description}")
            return True
        else:
            print(f"  Skipped: {description} (no change)")
            return False
    else:
        print(f"  Skipped: {description} (pattern not found)")
        return False

def main():
    powerinfer_dir = sys.argv[1] if len(sys.argv) > 1 else '/opt/powerinfer'

    print("=== Applying null pointer crash fixes to PowerInfer ===")
    print(f"Directory: {powerinfer_dir}")

    llama_cpp = os.path.join(powerinfer_dir, 'llama.cpp')
    ggml_cuda = os.path.join(powerinfer_dir, 'ggml-cuda.cu')

    fixes_applied = 0

    # === Fix 1: NULL check in add_tensor lambda ===
    print("\n=== Fix 1: NULL check in add_tensor lambda (llama.cpp) ===")

    pattern = r'if \(t->backend == GGML_BACKEND_GPU \|\| t->backend == GGML_BACKEND_GPU_SPLIT\)'
    replacement = 'if (t != nullptr && (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT))'

    if apply_fix(llama_cpp, pattern, replacement, "NULL check in add_tensor lambda"):
        fixes_applied += 1

    # === Fix 2: NULL check for KV cache tensors ===
    print("\n=== Fix 2: NULL check for KV cache tensors (llama.cpp) ===")

    # Only apply if not already patched
    with open(llama_cpp, 'r') as f:
        content = f.read()

    if 'cache.k == nullptr || cache.v == nullptr' not in content:
        pattern = r'(cache\.v = ggml_new_tensor_1d\(cache\.ctx, wtype, n_elements\);)\n(\s+ggml_set_name\(cache\.k,)'
        replacement = r'''\1

    // FIX: Check for allocation failure before using tensors
    if (cache.k == nullptr || cache.v == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate KV cache tensors\\n", __func__);
        ggml_free(cache.ctx);
        return false;
    }

\2'''

        if apply_fix(llama_cpp, pattern, replacement, "NULL check for KV cache tensor allocation"):
            fixes_applied += 1
    else:
        print("  Skipped: Already patched")

    # === Fix 3: Remove HIP self-pointer hack (first occurrence) ===
    print("\n=== Fix 3: Remove HIP self-pointer hack - up_gpu (llama.cpp) ===")

    # Use flexible regex pattern to handle whitespace variations
    pattern = r'#ifdef GGML_USE_HIPBLAS\s*\n// WARNING: THIS IS A HACK!\s*\n// if up_gpu->data is null\s*\n// inference fails when model exceeds 40B on rocm device\s*\n// so we just let up_gpu->data point to itself\s*\n\s*\n\s+up_gpu->data = up_gpu;\s*\n\s*\n#endif\s*'

    replacement = '''#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_CUBLAS)
    // FIX: Properly handle NULL up_gpu for gfx1151 and other ROCm targets
    // Instead of the dangerous self-pointer hack, skip GPU path when up_gpu is NULL
    if (up_gpu == nullptr) {
        // Fall through to CPU-only path
        out = ggml_mul_mat_idx(ctx, up, inp, idx, gpu_index);
        cb(out, full_name.c_str());
        return out;
    }
#endif

'''

    if apply_fix(llama_cpp, pattern, replacement, "Fixed up_gpu self-pointer hack"):
        fixes_applied += 1

    # === Fix 4: Remove HIP self-pointer hack (second occurrence) ===
    print("\n=== Fix 4: Remove HIP self-pointer hack - wt_gpu (llama.cpp) ===")

    pattern = r'#ifdef GGML_USE_HIPBLAS\s*\n// WARNING: THIS IS A HACK!\s*\n// if wt_gpu->data is null\s*\n// inference fails when model exceeds 40B on rocm device\s*\n// so we just let wt_gpu->data point to itself\s*\n\s*\n\s+wt_gpu->data = wt_gpu;\s*\n\s*\n#endif\s*'

    replacement = '''#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_CUBLAS)
    // FIX: Properly handle NULL wt_gpu for gfx1151 and other ROCm targets
    // Instead of the dangerous self-pointer hack, skip GPU path when wt_gpu is NULL
    if (wt_gpu == nullptr) {
        // Fall through to CPU-only path
        out = ggml_axpy(ctx, w_t, x, sparse_idx, gpu_index);
        cb(out, full_name.c_str());
        return out;
    }
#endif

'''

    if apply_fix(llama_cpp, pattern, replacement, "Fixed wt_gpu self-pointer hack"):
        fixes_applied += 1

    # === Fix 5: NULL check in ggml_cuda_assign_buffers_no_scratch ===
    print("\n=== Fix 5: NULL check in ggml_cuda_assign_buffers_no_scratch ===")

    with open(ggml_cuda, 'r') as f:
        content = f.read()

    if 'ggml_cuda_assign_buffers_no_scratch' in content and 'tensor == NULL' not in content.split('ggml_cuda_assign_buffers_no_scratch')[1][:200]:
        pattern = r'(void ggml_cuda_assign_buffers_no_scratch\(struct ggml_tensor \* tensor\) \{)\n(\s+ggml_cuda_assign_buffers_impl)'
        replacement = r'''\1
    // FIX: Add NULL check like ggml_cuda_assign_buffers() has
    if (tensor == NULL)
        return;
\2'''

        if apply_fix(ggml_cuda, pattern, replacement, "NULL check in ggml_cuda_assign_buffers_no_scratch"):
            fixes_applied += 1
    else:
        print("  Skipped: Already patched or pattern not found")

    # === Fix 6: Set g_device_count to 0 on failure ===
    print("\n=== Fix 6: Set g_device_count to 0 on failure ===")

    pattern = r'(if \(cudaGetDeviceCount\(&g_device_count\) != cudaSuccess\) \{)\n(\s+initialized = true;)\n(\s+g_cublas_loaded = false;)'
    replacement = r'''\1
            g_device_count = 0;  // FIX: Ensure device count is valid
\2
\3'''

    if apply_fix(ggml_cuda, pattern, replacement, "Set g_device_count = 0 on failure"):
        fixes_applied += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"=== Null pointer fixes complete: {fixes_applied} fixes applied ===")
    print("=" * 60)
    print("""
Summary of changes:
  1. add_tensor lambda now checks for NULL before accessing t->backend
  2. KV cache tensor allocation now validates success
  3. Dangerous HIP self-pointer hack (up_gpu) replaced with CPU fallback
  4. Dangerous HIP self-pointer hack (wt_gpu) replaced with CPU fallback
  5. ggml_cuda_assign_buffers_no_scratch now has NULL check
  6. g_device_count set to 0 when device enumeration fails

Rebuild with: cmake --build build --config Release
""")

    return 0 if fixes_applied > 0 else 1

if __name__ == '__main__':
    sys.exit(main())

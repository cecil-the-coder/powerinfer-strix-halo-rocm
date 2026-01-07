#!/bin/bash
# Apply gfx1151 (Strix Halo) compatibility patches to PowerInfer
# Based on kyuz0/amd-strix-halo-toolboxes approach

set -euo pipefail

POWERINFER_DIR="${1:-/opt/powerinfer}"

echo "=== Applying gfx1151 patches to PowerInfer ==="
echo "Directory: $POWERINFER_DIR"

# Find the vendors/hip.h file (could be in different locations)
HIP_HEADER=""
for candidate in \
    "$POWERINFER_DIR/smallthinker/ggml/src/ggml-cuda/vendors/hip.h" \
    "$POWERINFER_DIR/ggml/src/ggml-cuda/vendors/hip.h" \
    "$POWERINFER_DIR/vendors/hip.h"; do
    if [[ -f "$candidate" ]]; then
        HIP_HEADER="$candidate"
        break
    fi
done

if [[ -z "$HIP_HEADER" ]]; then
    echo "Warning: vendors/hip.h not found, checking for monolithic ggml-cuda.cu"
    # PowerInfer might use the old monolithic structure
    if [[ -f "$POWERINFER_DIR/ggml-cuda.cu" ]]; then
        echo "Found monolithic ggml-cuda.cu - applying inline patches"
        MONOLITHIC=1
    else
        echo "Error: Cannot find HIP-related files to patch"
        exit 1
    fi
else
    MONOLITHIC=0
    echo "Found HIP header: $HIP_HEADER"
fi

# Step 1: Patch vendors/hip.h if it exists
if [[ "$MONOLITHIC" == "0" ]]; then
    echo ""
    echo "Step 1: Patching vendors/hip.h..."

    # Check if already patched
    if grep -q "GGML_HIP_WARP_MASK" "$HIP_HEADER" 2>/dev/null; then
        echo "  Already patched (found GGML_HIP_WARP_MASK)"
    else
        # Backup original
        cp "$HIP_HEADER" "$HIP_HEADER.backup"

        # Find the __shfl_sync definition line
        SHFL_LINE=$(grep -n "^#define __shfl_sync" "$HIP_HEADER" | head -1 | cut -d: -f1 || echo "")

        if [[ -n "$SHFL_LINE" ]]; then
            # Create patched version
            {
                # Lines before __shfl_sync
                head -n $((SHFL_LINE - 1)) "$HIP_HEADER"

                # Insert our conditional block
                cat << 'PATCH'
// gfx1151 (Strix Halo) compatibility - use 64-bit warp masks for rocWMMA
#ifdef GGML_HIP_ROCWMMA_FATTN
#define GGML_HIP_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#else
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define GGML_HIP_WARP_MASK 0xFFFFFFFF
#endif
PATCH

                # Skip original __shfl_sync and __shfl_xor_sync lines, print rest
                tail -n +$((SHFL_LINE + 2)) "$HIP_HEADER"
            } > "$HIP_HEADER.tmp"

            mv "$HIP_HEADER.tmp" "$HIP_HEADER"
            echo "  Patched: Added GGML_HIP_WARP_MASK macro"
        else
            echo "  Warning: Could not find __shfl_sync definition to patch"
        fi
    fi
fi

# Step 2: Replace hardcoded warp masks in CUDA files
echo ""
echo "Step 2: Replacing hardcoded warp masks..."

# Find CUDA directories
CUDA_DIRS=()
for dir in \
    "$POWERINFER_DIR/smallthinker/ggml/src/ggml-cuda" \
    "$POWERINFER_DIR/ggml/src/ggml-cuda" \
    "$POWERINFER_DIR"; do
    if [[ -d "$dir" ]]; then
        CUDA_DIRS+=("$dir")
    fi
done

MODIFIED_COUNT=0
for dir in "${CUDA_DIRS[@]}"; do
    # Find all .cu and .cuh files
    while IFS= read -r -d '' file; do
        # Check if file contains hardcoded masks (but not already patched)
        if grep -q "0xFFFFFFFF\|0xffffffff" "$file" 2>/dev/null; then
            if ! grep -q "GGML_HIP_WARP_MASK" "$file" 2>/dev/null; then
                # Backup and patch
                cp "$file" "$file.backup"
                sed -i 's/0xFFFFFFFF/GGML_HIP_WARP_MASK/g; s/0xffffffff/GGML_HIP_WARP_MASK/g' "$file"
                MODIFIED_COUNT=$((MODIFIED_COUNT + 1))
                echo "  Patched: $(basename "$file")"
            fi
        fi
    done < <(find "$dir" -maxdepth 2 \( -name "*.cu" -o -name "*.cuh" \) -print0 2>/dev/null)
done

echo "  Modified $MODIFIED_COUNT files"

# Step 3: Copy hip_shfl_fix.h if provided
echo ""
echo "Step 3: Installing hip_shfl_fix.h..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/hip_shfl_fix.h" ]]; then
    for dir in "${CUDA_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            cp "$SCRIPT_DIR/hip_shfl_fix.h" "$dir/"
            echo "  Installed: $dir/hip_shfl_fix.h"
            break
        fi
    done
else
    echo "  Warning: hip_shfl_fix.h not found in script directory"
fi

echo ""
echo "=== gfx1151 patches applied successfully ==="
echo ""
echo "Build with:"
echo "  cmake -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1151 ..."
echo ""
echo "Runtime environment:"
echo "  export HSA_OVERRIDE_GFX_VERSION=11.5.1"
echo "  export ROCBLAS_USE_HIPBLASLT=1"

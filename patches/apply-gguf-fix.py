#!/usr/bin/env python3
"""
Fix gguf.Keys.Split.VRAM_CAPACITY compatibility issue in PowerInfer

The powerinfer-py module expects a custom gguf library with the VRAM_CAPACITY key,
but the standard PyPI gguf package doesn't have this. This patch adds the missing
constant to make the export_split.py work with the standard gguf package.
"""

import os
import sys

def main():
    powerinfer_dir = sys.argv[1] if len(sys.argv) > 1 else '/opt/powerinfer'

    export_split_path = os.path.join(powerinfer_dir, 'powerinfer-py', 'powerinfer', 'export_split.py')

    if not os.path.exists(export_split_path):
        print(f"  ERROR: File not found: {export_split_path}")
        return 1

    with open(export_split_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'VRAM_CAPACITY_KEY' in content:
        print("  Already patched: export_split.py")
        return 0

    # Add the missing constant definition and fix the usage
    patch = '''import argparse
import pickle
import sys
from gguf.constants import GGMLQuantizationType
from gguf.gguf_writer import GGUFWriter
import torch
from pathlib import Path
import os
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
import struct
import numpy as np
import re

# PowerInfer custom key - not in standard gguf package
VRAM_CAPACITY_KEY = "split.vram_capacity"
'''

    # Replace the imports and add our constant
    old_imports = '''import argparse
import pickle
import sys
from gguf.constants import GGMLQuantizationType
from gguf.gguf_writer import GGUFWriter
import torch
from pathlib import Path
import os
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
import struct
import numpy as np
import re'''

    if old_imports in content:
        content = content.replace(old_imports, patch)
        # Also fix the usage of the constant
        content = content.replace(
            'gguf_out.add_uint64(gguf.Keys.Split.VRAM_CAPACITY, vram_capacity)',
            'gguf_out.add_uint64(VRAM_CAPACITY_KEY, vram_capacity)'
        )

        with open(export_split_path, 'w') as f:
            f.write(content)
        print("  Applied: gguf.Keys.Split.VRAM_CAPACITY fix to export_split.py")
        return 0
    else:
        print("  Skipped: Pattern not found in export_split.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())

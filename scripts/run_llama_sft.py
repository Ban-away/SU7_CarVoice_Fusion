#!/usr/bin/env python
"""Wrapper to run LLaMA-Factory SFT after patching numpy compatibility."""
import numpy as np
for name in ('long','ulong','int','float','complex','unicode','object','bool','str'):
    if not hasattr(np, name):
        setattr(np, name, getattr(np, name.replace('ulong','uint64').replace('long','int64')
            .replace('int','int32').replace('float','float64').replace('complex','complex128')
            .replace('unicode','str_').replace('object','object_').replace('bool','bool_')
            .replace('str','str_'), 'int64')))

# Fix mapping
_map = {'long': np.int64, 'ulong': np.uint64, 'int': np.int32, 'float': np.float64,
        'complex': np.complex128, 'unicode': np.str_, 'object': np.object_, 'bool': np.bool_, 'str': np.str_}
for k, v in _map.items():
    if not hasattr(np, k):
        setattr(np, k, v)

# Now run llamafactory
import subprocess, sys
sys.exit(subprocess.run(['llamafactory-cli', 'train', 'configs/sft.yaml'],
    env={**__import__('os').environ, 'HF_HUB_OFFLINE': '1'}).returncode)

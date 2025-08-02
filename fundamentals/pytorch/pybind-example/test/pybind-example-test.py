#!/usr/bin/env python3
import sys
import os

# Add build directory
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

try:
    import pybind_example
    print("✓ Import successful!")

    print("\nAvailable in module:")
    for attr in dir(pybind_example):
        if not attr.startswith('_'):
            print(f"  {attr}: {type(getattr(pybind_example, attr))}")

    # Try common function names
    if hasattr(pybind_example, 'add'):
        result = pybind_example.add(10, 20)
        print(f"\nadd(10, 20) = {result}")

except ImportError as e:
    print(f"Import failed: {e}")
    print("Make sure the .so file exists in ./build/")

#!/usr/bin/env python3
import argparse
import json
import struct

DEFAULT_MODEL = "models/Qwen2.5-0.5B-Instruct-gptq-4bit/model.safetensors"

def read_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    args = parser.parse_args()

    header = read_header(args.model)
    metadata = header.pop("__metadata__", None)
    if metadata:
        print(f"metadata: {metadata}\n")

    for name, info in sorted(header.items()):
        print(f"{name:<60} {info['dtype']:<6} {info['shape']}")

    print(f"\ntotal tensors: {len(header)}")


if __name__ == "__main__":
    main()

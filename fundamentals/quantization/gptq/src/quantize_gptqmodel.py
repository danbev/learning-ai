#!/usr/bin/env python3
"""Quantize a small model to 4-bit GPTQ with GPTQModel and inspect the result.

The output directory is a regular HF checkpoint with quant_method "gptq", which
is what llama.cpp's convert_hf_to_gguf.py handles with dequant_gptq().
"""
import argparse
import json
from pathlib import Path

from gptqmodel import GPTQConfig, GPTQModel
from safetensors import safe_open

from calibration import load_calibration_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="models/Qwen2.5-0.5B-Instruct-gptq-4bit")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    quant_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,   # input columns that share one scale/zero-point
        sym=True,                     # symmetric: zero-point is the midpoint (8 for 4-bit)
        desc_act=False,               # no act-order, so g_idx is just column // group_size
        act_group_aware=False,        # GPTQModel-specific reordering, off to match plain GPTQ
        format="gptq",                # v1 on-disk layout (the default), what llama.cpp expects
    )

    calibration = load_calibration_texts(args.n_samples)
    print(f"calibration: {len(calibration)} texts")

    model = GPTQModel.load(args.model, quant_config)
    model.quantize(calibration, batch_size=args.batch_size)
    model.save(args.out)

    show_result(Path(args.out))


def show_result(out: Path):
    config = json.loads((out / "config.json").read_text())
    print("\nquantization_config:")
    print(json.dumps(config.get("quantization_config"), indent=2))

    print("\nlayer 0 down_proj tensors:")
    for file in sorted(out.glob("*.safetensors")):
        with safe_open(file, framework="pt") as f:
            for name in sorted(f.keys()):
                if "layers.0.mlp.down_proj" in name:
                    s = f.get_slice(name)
                    print(f"  {name:50} {s.get_dtype():5} {s.get_shape()}")


if __name__ == "__main__":
    main()

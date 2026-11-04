#!/usr/bin/env python3
"""Quantize a small HF model with NVIDIA ModelOpt and export a HF checkpoint.

Flow:
  1. load the bf16 model
  2. mtq.quantize() inserts TensorQuantizer modules (input/weight/output) into
     each nn.Linear and runs the forward_loop over calibration data so the
     quantizers can record amax (max abs value) statistics.
  3. export_hf_checkpoint() folds the quantizers into real packed weights +
     scales and writes config.json with a "quantization_config" (quant_method
     "modelopt") and a hf_quant_config.json.

The default is NVFP4 (W4A4: FP4 E2M1 values, one FP8 E4M3 scale per 16
elements, plus a per-tensor FP32 global scale).
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import ensure_local_checkpoint, export_hf_checkpoint
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

QFORMATS = {
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,      # W4A4 FP4, block 16, FP8 scales
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG, # NVFP4 + AWQ-lite pre-scaling
    "w4a16_nvfp4": mtq.W4A16_NVFP4_CFG,  # NVFP4 weights only
    "fp8": mtq.FP8_DEFAULT_CFG,          # W8A8 FP8 E4M3, per-tensor
    "int4_awq": mtq.INT4_AWQ_CFG,        # W4A16 INT4, block 128, AWQ
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG, # W8A8 INT8 SmoothQuant
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--qformat", choices=QFORMATS, default="nvfp4")
    parser.add_argument("--out", default=None, help="default: models/<model>-<qformat>")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out or f"models/{args.model.split('/')[-1]}-{args.qformat}")

    _, local_path = ensure_local_checkpoint(args.model)

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    calib = load_calibration(tokenizer, args.n_samples, args.max_seq_len)

    def forward_loop(m):
        # ModelOpt only needs activations to flow through the model; the
        # quantizers observe amax during these forward passes.
        for i in range(0, len(calib), args.batch_size):
            batch = tokenizer(calib[i : i + args.batch_size], return_tensors="pt",
                              padding=True, truncation=True, max_length=args.max_seq_len)
            m(**batch.to(m.device))

    cfg = copy.deepcopy(QFORMATS[args.qformat])
    with torch.no_grad():
        model = mtq.quantize(model, cfg, forward_loop)

    # Shows every TensorQuantizer, its format and calibrated amax.
    mtq.print_quant_summary(model)

    generate(model, tokenizer, "What is the capital of Sweden?")

    with torch.no_grad():
        export_hf_checkpoint(model, export_dir=out)

    show_result(out)


def load_calibration(tokenizer, n_samples, max_seq_len):
    """Chat formatted calibration samples (same dataset as ../compressed-tensors)."""
    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
    return [tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in ds]


def generate(model, tokenizer, prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    print(f"\n[fake-quant generate] {prompt!r} -> "
          f"{tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)!r}\n")


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
                    print(f"  {name:50} {s.get_dtype():8} {s.get_shape()}")


if __name__ == "__main__":
    main()

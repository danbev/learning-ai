import torch
from transformers import AutoConfig, AutoModelForCausalLM

path = "./model"

config = AutoConfig.from_pretrained(path, trust_remote_code=True)
print("Loaded config:", config)

model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    low_cpu_mem_usage=False,
    device_map=None,
    dtype=torch.float32,
)
bad = [n for n,p in model.named_parameters() if p.device.type == "meta"]
bad += [f"(buffer) {n}" for n,b in model.named_buffers() if b.device.type == "meta"]
print("Left on meta:", bad)

model.eval()

input_ids = torch.randint(0, config.vocab_size, (1, 8))
out = model(input_ids)
print("Logits shape:", out.logits.shape)  # (1, 8, vocab_size)



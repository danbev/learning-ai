from vllm import LLM, SamplingParams
import torch

class LogitsPrinter:
    def __call__(self, token_ids, logits):
        print(f"\n=== Logits at step {len(token_ids)} ===")
        print(f"Logits shape: {logits.shape}")
        print(f"Min logit: {logits.min().item():.4f}")
        print(f"Max logit: {logits.max().item():.4f}")

        # Get top-k tokens
        top_k = 5
        top_values, top_indices = torch.topk(logits, top_k)
        print(f"\nTop {top_k} tokens:")
        for idx, (val, token_id) in enumerate(zip(top_values, top_indices)):
            print(f"  {idx+1}. Token {token_id}: logit={val.item():.4f}")

        return logits  # Must return logits unchanged

llm = LLM(model="path/to-model", trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=10,
    logits_processors=[LogitsPrinter()]
)

outputs = llm.generate(["Hello, my name is"], sampling_params)


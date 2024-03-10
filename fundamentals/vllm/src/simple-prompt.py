from vllm import LLM, SamplingParams

methods = [method for method in dir(LLM) if callable(getattr(LLM, method)) and not method.startswith("_")]
print(methods)

prompts = [
    "What is the captial of Sweden?",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

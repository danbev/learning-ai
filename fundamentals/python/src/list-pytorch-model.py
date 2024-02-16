import torch

# Load the model or tensor
model = torch.load('/home/danielbevenius/work/ai/llava-v1.5-7b/llava.projector')

# If it's a model's state_dict
if isinstance(model, dict):
    for key in model:
        print(f"{key}: {model[key].size()}")

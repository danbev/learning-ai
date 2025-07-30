import torch.nn as nn
import torch
import inspect

print(inspect.getsource(nn.Module.__init__))
#print(inspect.getsource(nn.Module.__setattr__))
print(inspect.getsource(nn.Module.__getattr__))

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing SimpleModule")

        self.layer1 = nn.Linear(10, 5)  # 10 inputs -> 5 outputs
        #print("Layer 1:", self.layer1.weight)
        # Automatically get a bias term.
        print("Has bias:", self.layer1.bias is not None)  # True
        print("Bias shape:", self.layer1.bias.shape)      # torch.Size([5])
        print("Bias values:", self.layer1.bias)           # tensor([0., 0., 0., 0., 0.])

    def forward(self, x):
        print("Forward pass called")
        return 18

module = SimpleModule()
print(module)

output = module(1)

print("nn.Module __dict__ contents:")
for key, value in module.__dict__.items():
    print(f"  {key}: {type(value)} = {value}")

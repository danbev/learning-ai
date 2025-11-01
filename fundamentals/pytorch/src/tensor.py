import torch

def print_tensor(t: torch.Tensor):
    print(t, type(t), t.dtype, t.device, t.requires_grad)

def main():
    print("Tensor creation examples")
    # We can create a tensor from a Python list:
    t = torch.tensor([1, 2])
    print_tensor(t)

    # From a list of lists:
    t = torch.tensor([[1, 2], [3, 4]])
    print_tensor(t)

    t = torch.tensor(5.0)
    print_tensor(t)

    t = torch.tensor([0, 0, 0],   # Initial data
             dtype=None,      # Data type (torch.float32, torch.int64, etc.)
             device=None,     # Device ('cpu', 'cuda', 'cuda:0', etc.)
             requires_grad=False)  # Track gradients for autograd
    print_tensor(t)

    t = torch.range(0, 10, 2)  # Start, end, how big each step should be (we know the step size here)
    print_tensor(t)

    t = torch.linspace(0, 10, 5)  # Start, end, number of steps (we know the number of points needed)
    print_tensor(t)


if __name__ == "__main__":
    main()


import torch

def main():
    print("Squeeze Tensor Example")
    t = torch.tensor(
        [ # first dimension just has one element
            [1, 2, 3] # second dimension has three elements
        ] # this would be shape [3, 1] in ggml.
    )
    # So what sequeeze does it it removes the empty dimensions.
    print(t.shape)
    t = t.squeeze()
    print(t.shape)

    # And we can also add a dimension back in.
    t = t.unsqueeze(0)  # Add a dimension at index 0
    print(t.shape)

    t = t.unsqueeze(0)  # Add a dimension at index 0
    print(t.shape)

# When a script is run directly the special variable __main__ is set to "__main__"
print(f"__name__ = {__name__}")
if __name__ == "__main__":
    main()

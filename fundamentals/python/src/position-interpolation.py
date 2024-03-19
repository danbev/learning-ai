def create_extended_position_mapping(original_length, new_length):
    mapping = {}
    for new_index in range(1, new_length + 1):
        # // is integer division in python
        # 
        original_index = (new_index + 1) // 2
        mapping[new_index] = original_index

    return mapping

original_length = 512
new_length = 1024

extended_position_mapping = create_extended_position_mapping(original_length, new_length)

for i in range(1, 21):
    print(f"New Position {i} -> Mapped from Original Position {extended_position_mapping[i]}")

# This looked super strange to me at first, but it makes sense if you think
# about it. The next step is to fine-tuning the model on longer sequences usin
# this new mapping. This way we help the model adapt to the fact that two
# consecutive positions now share the same embedding. The fine-tuning process
# allows the model to learn how to interpret and handle these "stretched"
# positional embeddings in the context of longer sequences.

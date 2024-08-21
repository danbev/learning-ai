import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_printoptions(sci_mode=False, precision=4)

gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

with torch.no_grad():
    inputs = gpt2_tokenizer("Dan loves icecream", return_tensors="pt", add_special_tokens=False)
    attentions = gpt2(inputs.input_ids, output_attentions=True).attentions

    # get the attention scores computed by the first layer
    # for the first input sequence in the batch
    first_layer_attentions = attentions[0][0]

    # print attention scores from the first head
    print("GPT2 Attention Scores (Head 1):")
    print(first_layer_attentions[0])

    inputs = gpt2_tokenizer("Dan loves icecream but", return_tensors="pt", add_special_tokens=False)
    attentions = gpt2(inputs.input_ids, output_attentions=True).attentions

    # get the attention scores computed by the first layer
    # for the first input sequence in the batch
    first_layer_attentions = attentions[0][0]

    # print attention scores from the first head
    print("GPT2 Attention Scores (Head 1):")
    print(first_layer_attentions[0])

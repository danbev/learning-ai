text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
# import torch
import torch
import pandas as pd

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

outputs = classifier(text)
#outputs = classifier(text, model="distilbert-base-uncased-finetuned-sst-2-english")
print(pd.DataFrame(outputs))

emotions = load_dataset("emotion")
print(emotions)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

print(f'(tokens: {tokenize(emotions["train"][:2])}')

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(type(device))
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"
print(f'text: {text}')
inputs = tokenizer(text, return_tensors="pt")
print(f'input.items: {inputs.items()}')
print(f"Input tensor shape: {inputs['input_ids'].size()}")

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(f'[batch_size, nr_tokens, hidden_dim]: {outputs.last_hidden_state.size()}')

# It was a little connfusing to me that the the number of tokens was 6 when we
# only have 4 words in the sentence. The reason for this is that the tokenizer
# also adds special tokens to the input. In this case, the tokenizer adds a
# special token for the beginning of the sentence ([CLS]) and a special token
# for the end of the sentence ([SEP]).
print(inputs['input_ids'][0])
#print(inputs[0].input_ids)
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

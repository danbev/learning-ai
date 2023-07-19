# Tokens section
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
# A very simple way of tokenizing a string:
print(f'char tokens: {tokenized_text}')
# But for these tokens to be used in a model they need to be number/integers
# so we need to convert them to integers.
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(f'token2idx: {token2idx}')

input_ids = [token2idx[token] for token in tokenized_text]
print(f'Tokens: {input_ids}')

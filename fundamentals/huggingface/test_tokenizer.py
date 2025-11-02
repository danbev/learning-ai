from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer", trust_remote_code=True)

print("✓ Successfully loaded with AutoTokenizer!")
print(f"Type: {type(tokenizer)}")
print(f"Vocab size: {len(tokenizer)}")
print()

# Test encoding
text = "Hello, World!"
encoded = tokenizer.encode(text)
print(f"Text: '{text}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{tokenizer.decode(encoded)}'")
print()

# Test special tokens
print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"UNK token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
print(f"BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

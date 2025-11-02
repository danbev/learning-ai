from tokenizer.tokenization_custom import CustomTokenizer

# Create tokenizer
tokenizer = CustomTokenizer(vocab_size=100)

# Set auto_map as an attribute
tokenizer.auto_map = {
    "AutoTokenizer": ["tokenization_custom.CustomTokenizer", None]
}
tokenizer.save_pretrained("tokenizer")

print("Tokenizer saved to ./tokenizer")
print(f"Vocab size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.all_special_tokens}")

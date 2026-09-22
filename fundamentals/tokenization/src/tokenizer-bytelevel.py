import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.processors import ByteLevel as ByteLevelPost

# Vocab mapping pre-tokenized byte-level strings directly to IDs
vocab = {
    "hello": 0,
    "Ġworld": 1,
    "<unk>": 2,
}

tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)

# Attach ByteLevel post-processor with trim_offsets=False
tokenizer.post_processor = ByteLevelPost(
    trim_offsets=False,
    use_regex=True
)

text = "hello world"
enc = tokenizer.encode(text)

print("=== trim_offsets=False ===")
print("Tokens: ", enc.tokens)
print("IDs:    ", enc.ids)
print("Offsets:", enc.offsets)
print("Slice 1:", repr(text[enc.offsets[1][0]:enc.offsets[1][1]]))

# Change trim_offsets to True to observe the difference
tokenizer.post_processor = ByteLevelPost(
    trim_offsets=True,
    use_regex=True
)

enc_trimmed = tokenizer.encode(text)
print("\n=== trim_offsets=True ===")
print("Tokens: ", enc_trimmed.tokens)
print("IDs:    ", enc_trimmed.ids)
print("Offsets:", enc_trimmed.offsets)
print("Slice 1:", repr(text[enc_trimmed.offsets[1][0]:enc_trimmed.offsets[1][1]]))

serialized = json.loads(tokenizer.to_str())
print("\nSerialized post_processor in tokenizer.json:")
print(json.dumps(serialized["post_processor"], indent=2))

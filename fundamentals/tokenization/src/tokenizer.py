import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

vocab = {"hello": 0, "world": 1, "<s>": 2, "</s>": 3}
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

enc = tokenizer.encode("hello world")
print("Without post-processor:")
print("Raw tokens:", enc.tokens)
print("Raw IDs:   ", enc.ids)
# So this only produces the actually token for "hello world" and nothing else.

# The core tokenizer only knows how to split the input into tokens. The
# post-processor is what enables us to add special tokens.

# The post_processor takes the tokens produced by the tokenizer and assigns the
# the the varialbe $A.
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",   # replaces $A with <s> $A </s>
    pair="<s> $A </s> $B </s>",
    special_tokens=[
        ("<s>", 2),   # When we encounter <s> emit token id 2.
        ("</s>", 3),  # When we encouter </s> emit token id 3.
                      # This is not related to the vocab above.
    ],
)

# With Post-Processor (add_special_tokens=True)
enc_special = tokenizer.encode("hello world", add_special_tokens=True)
print("With TemplateProcessing (add_special_tokens=True):")
print("Tokens:", enc_special.tokens)
print("IDs:   ", enc_special.ids)

# With Post-Processor (add_special_tokens=False)
enc_no_special = tokenizer.encode("hello world", add_special_tokens=False)
print("\nWith TemplateProcessing (add_special_tokens=False):")
print("Tokens:", enc_no_special.tokens)
print("IDs:   ", enc_no_special.ids)

# Inspect how this serializes into tokenizer.json
serialized = json.loads(tokenizer.to_str())
print("\nSerialized post_processor in tokenizer.json:")
print(json.dumps(serialized["post_processor"], indent=2))

print(tokenizer.token_to_id("<s>"))

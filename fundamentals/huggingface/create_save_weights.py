from model.configuration_model import ModelConfig
from model.modeling_model      import ModelLM

config = ModelConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, tie_word_embeddings=True)
config.auto_map = {
    "AutoConfig": "configuration_model.ModelConfig",
    "AutoModel": "modeling_model.ModelLM",
    "AutoModelForCausalLM": "modeling_model.ModelLM"
}
model = ModelLM(config)

model.save_pretrained("model", safe_serialization=True)
print("Saved to ./model")


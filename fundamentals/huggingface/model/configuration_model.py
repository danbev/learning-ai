from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "model"

    def __init__(
        self,
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

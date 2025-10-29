import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from .configuration_model import ModelConfig

class ModelLM(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = "backbone"
    ## Use the same tensor for input embeddings and output embeddings
    _tied_weights_keys = ["lm_head.weight", "backbone.embed.weight"]


    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.backbone = nn.Module()
        self.backbone.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.backbone.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embed

    def set_input_embeddings(self, value):
        self.backbone.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_emb):
        self.lm_head = new_emb

    def tie_weights(self):
        out_emb = self.get_output_embeddings()   # lm_head (Linear)
        in_emb  = self.get_input_embeddings()    # Embedding

        # If either side is missing, do nothing
        if out_emb is None or in_emb is None:
            return

        out_w = out_emb.weight
        in_w  = in_emb.weight

        if in_w.device.type == "meta" and out_w.device.type != "meta":
            # IMPORTANT: rebind the Parameter, not just copy data
            in_emb.weight = out_w
            return

        if out_w.device.type == "meta" and in_w.device.type != "meta":
            out_emb.weight = in_w
            return

        # Default HF behavior (ties by reference or clones as needed)
        self._tie_or_clone_weights(out_emb, in_emb)


    def forward(self, input_ids=None, labels=None, **kwargs):
        # input_ids: (batch, seq_len)
        x = self.backbone.embed(input_ids)       # (B, T, H)
        x = self.backbone.mlp(x)                 # (B, T, H)
        logits = self.lm_head(x)                 # (B, T, V)

        loss = None
        if labels is not None:
            # classic language-model loss with next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size),
                                         shift_labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)


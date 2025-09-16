### Converting notes for Qwen/Qwen2.5-VL-3B-Instruct
This is a multi-modal [model](https://github.com/QwenLM/Qwen2.5-VL). llama.cpp
currently has support for 

First clone the model reposition from Hugging Face:
```console
$ git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
```

To be able to convert this model we need to update the gguf-py scripts to
support it. There currently is support for Qwen2-VL so we can start with using
the same tensor names and conversion as a starting point:
```console
diff --git a/convert_hf_to_gguf.py b/convert_hf_to_gguf.py
index 63b54a9c..d7debf27 100755
--- a/convert_hf_to_gguf.py
+++ b/convert_hf_to_gguf.py
@@ -2211,7 +2211,7 @@ class Qwen2Model(Model):
                 self.gguf_writer.add_rope_scaling_orig_ctx_len(self.hparams["rope_scaling"]["original_max_position_embeddings"])
 
 
-@Model.register("Qwen2VLForConditionalGeneration")
+@Model.register("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration")
 class Qwen2VLModel(Model):
     model_arch = gguf.MODEL_ARCH.QWEN2VL
 
diff --git a/gguf-py/gguf/constants.py b/gguf-py/gguf/constants.py
index 8fe84df2..d5e64ad4 100644
--- a/gguf-py/gguf/constants.py
+++ b/gguf-py/gguf/constants.py
@@ -242,6 +242,7 @@ class MODEL_ARCH(IntEnum):
     QWEN2            = auto()
     QWEN2MOE         = auto()
     QWEN2VL          = auto()
+    QWEN25VL         = auto()
     PHI2             = auto()
     PHI3             = auto()
     PHIMOE           = auto()
@@ -429,6 +430,7 @@ MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
     MODEL_ARCH.QWEN2:            "qwen2",
     MODEL_ARCH.QWEN2MOE:         "qwen2moe",
     MODEL_ARCH.QWEN2VL:          "qwen2vl",
+    MODEL_ARCH.QWEN25VL:         "qwen25vl",
     MODEL_ARCH.PHI2:             "phi2",
     MODEL_ARCH.PHI3:             "phi3",
     MODEL_ARCH.PHIMOE:           "phimoe",
@@ -870,6 +872,20 @@ MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
         MODEL_TENSOR.FFN_DOWN,
         MODEL_TENSOR.FFN_UP,
     ],
+    MODEL_ARCH.QWEN25VL: [
+        MODEL_TENSOR.TOKEN_EMBD,
+        MODEL_TENSOR.OUTPUT_NORM,
+        MODEL_TENSOR.OUTPUT,
+        MODEL_TENSOR.ATTN_NORM,
+        MODEL_TENSOR.ATTN_Q,
+        MODEL_TENSOR.ATTN_K,
+        MODEL_TENSOR.ATTN_V,
+        MODEL_TENSOR.ATTN_OUT,
+        MODEL_TENSOR.FFN_NORM,
+        MODEL_TENSOR.FFN_GATE,
+        MODEL_TENSOR.FFN_DOWN,
+        MODEL_TENSOR.FFN_UP,
+    ],
     MODEL_ARCH.QWEN2MOE: [
         MODEL_TENSOR.TOKEN_EMBD,
         MODEL_TENSOR.OUTPUT_NORM,

```




 gguf-dump --json --json-array --no-tensors converted/Qwen2.5-0.5B-Instruct-nvfp4_mixed.gguf \
  | jq -c '.metadata as $m
           | [$m["general.tensor_extra.name"].value, $m["general.tensor_extra.prec_a4"].value]
           | transpose[] | {name: .[0], prec_a4: .[1]}'

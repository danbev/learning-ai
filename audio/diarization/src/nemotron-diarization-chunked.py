from nemo.collections.asr.models import SortformerEncLabelModel

diar_model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")
diar_model.eval()

diar_model.sortformer_modules.chunk_len = 104
diar_model.sortformer_modules.chunk_right_context = 8
diar_model.sortformer_modules.fifo_len = 80
diar_model.sortformer_modules.spkcache_update_period = 320
diar_model._check_streaming_parameters()


predicted_segments = diar_model.diarize(audio=["samples/ami_en2002d_2132.wav"], batch_size=1)

for segment in sorted(predicted_segments[0], key=lambda s: float(s.split()[0])):
    print(segment)


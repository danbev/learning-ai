## Voice Activity Detection (VAD)
Also known as as speech activity detection (SAD) or speech detection.

Now, keep in mind that this is different than Automatic Speech Recognition (ASR)
which is the process of converting speech into text. VAD is used to determine
whether a segment of audio contains speech or not. It is often used as a
preprocessing step in ASR systems to filter out non-speech segments and reduce
the amount of data that needs to be processed. So it would be like a preprocessor
of an audio signal to remove silence or non-speech segments.
For example ASR systems may struggle with long periods of silence or noise, and
can output strange results if they are not filtered out.

So VAD should tell speech apart from noise and silence. It could be used in
mobil or IoT devices to detace human speech for example.
So the input is a small audio segment/chunk and the output is a probability
that this chunk contains speech.

### Silero-VAD
github: https://github.com/snakers4/silero-vad

The model that Silero-VAD has is not publicly available yet. I found this
discussion:
https://github.com/snakers4/silero-vad/discussions/371

But they do provide their model in two formats, one PyTorch JIT (Just In Time)
format and one in ONNX format. So we have access to the models, but the actual
PyTorch code for the model does not seem to be available which made it a little
difficult for me to figure out if the whisper.cpp version was correct or not.

To find out more about the model I used the following script:
```console
(venv) $ python src/inspect-jit.py
RecursiveScriptModule(
  original_name=VADRNNJIT
  (stft): RecursiveScriptModule(
    original_name=STFT
    (padding): RecursiveScriptModule(original_name=ReflectionPad1d)
  )
  (encoder): RecursiveScriptModule(
    original_name=Sequential
    (0): RecursiveScriptModule(
      original_name=SileroVadBlock
      (se): RecursiveScriptModule(original_name=Identity)
      (activation): RecursiveScriptModule(original_name=ReLU)
      (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
    )
    (1): RecursiveScriptModule(
      original_name=SileroVadBlock
      (se): RecursiveScriptModule(original_name=Identity)
      (activation): RecursiveScriptModule(original_name=ReLU)
      (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
    )
    (2): RecursiveScriptModule(
      original_name=SileroVadBlock
      (se): RecursiveScriptModule(original_name=Identity)
      (activation): RecursiveScriptModule(original_name=ReLU)
      (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
    )
    (3): RecursiveScriptModule(
      original_name=SileroVadBlock
      (se): RecursiveScriptModule(original_name=Identity)
      (activation): RecursiveScriptModule(original_name=ReLU)
      (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
    )
  )
  (decoder): RecursiveScriptModule(
    original_name=VADDecoderRNNJIT
    (rnn): RecursiveScriptModule(original_name=LSTMCell)
    (decoder): RecursiveScriptModule(
      original_name=Sequential
      (0): RecursiveScriptModule(original_name=Dropout)
      (1): RecursiveScriptModule(original_name=ReLU)
      (2): RecursiveScriptModule(original_name=Conv1d)
      (3): RecursiveScriptModule(original_name=Sigmoid)
    )
  )
)
Module: RecursiveScriptModule
Parameter: _model.encoder.0.reparam_conv.weight, Shape: torch.Size([128, 129, 3])
Parameter: _model.encoder.0.reparam_conv.bias, Shape: torch.Size([128])
Parameter: _model.encoder.1.reparam_conv.weight, Shape: torch.Size([64, 128, 3])
Parameter: _model.encoder.1.reparam_conv.bias, Shape: torch.Size([64])
Parameter: _model.encoder.2.reparam_conv.weight, Shape: torch.Size([64, 64, 3])
Parameter: _model.encoder.2.reparam_conv.bias, Shape: torch.Size([64])
Parameter: _model.encoder.3.reparam_conv.weight, Shape: torch.Size([128, 64, 3])
Parameter: _model.encoder.3.reparam_conv.bias, Shape: torch.Size([128])
Parameter: _model.decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
Parameter: _model.decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
Parameter: _model.decoder.rnn.bias_ih, Shape: torch.Size([512])
Parameter: _model.decoder.rnn.bias_hh, Shape: torch.Size([512])
Parameter: _model.decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
Parameter: _model.decoder.decoder.2.bias, Shape: torch.Size([1])
Buffer: _model.stft.forward_basis_buffer, Shape: torch.Size([258, 1, 256])
```
(I've removed the 8kHz model for now as we are not using it).

Now, the problem I had was that I could not find a way to print out intermediate
tensor values, like printing the output of the STFT layer or any of the others
to verify that out whisper.cpp implementation was correct or not. I spent a good
chunk of time trying to do so before basically giving up. But I need something
to compare with so by inspecting the model more closely using the same script
as above I was able to get information about each part of the model.
For example in the `_model.stft` we can find:
```console
Transform method code:
def transform_(self,
    input_data: Tensor) -> Tuple[Tensor, Tensor]:
  padding = self.padding
  input_data0 = torch.unsqueeze((padding).forward(input_data, ), 1)
  forward_basis_buffer = self.forward_basis_buffer
  hop_length = self.hop_length
  forward_transform = torch.conv1d(input_data0, forward_basis_buffer, None, [hop_length], [0])
  filter_length = self.filter_length
  _0 = torch.add(torch.div(filter_length, 2), 1)
  cutoff = int(_0)
  _1 = torch.slice(torch.slice(forward_transform), 1, None, cutoff)
  real_part = torch.to(torch.slice(_1, 2), 6)
  _2 = torch.slice(torch.slice(forward_transform), 1, cutoff)
  imag_part = torch.to(torch.slice(_2, 2), 6)
  _3 = torch.add(torch.pow(real_part, 2), torch.pow(imag_part, 2))
  magnitude = torch.sqrt(_3)
  phase = torch.atan2(ops.prim.data(imag_part), ops.prim.data(real_part))
  return (magnitude, phase)

Module: RecursiveScriptModule
Buffer: forward_basis_buffer, Shape: torch.Size([258, 1, 256])
Available methods: ['T_destination', '__call__', 'add_module', 'apply', 'bfloat16', 'buffers', 'call_super_init', 'children', 'code', 'code_with_constants', 'compile', 'cpu', 'cuda', 'define', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'forward_basis_buffer', 'forward_magic_method', 'get_buffer', 'get_debug_state', 'get_extra_state', 'get_parameter', 'get_submodule', 'graph', 'graph_for', 'half', 'inlined_graph', 'ipu', 'load_state_dict', 'modules', 'mtia', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'original_name', 'padding', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_load_state_dict_pre_hook', 'register_module', 'register_parameter', 'register_state_dict_post_hook', 'register_state_dict_pre_hook', 'requires_grad_', 'save', 'save_to_buffer', 'set_extra_state', 'set_submodule', 'share_memory', 'state_dict', 'to', 'to_empty', 'train', 'transform_', 'type', 'xpu', 'zero_grad']
Transform method code:
def transform_(self,
    input_data: Tensor) -> Tuple[Tensor, Tensor]:
  padding = self.padding
  input_data0 = torch.unsqueeze((padding).forward(input_data, ), 1)
  forward_basis_buffer = self.forward_basis_buffer
  hop_length = self.hop_length
  forward_transform = torch.conv1d(input_data0, forward_basis_buffer, None, [hop_length], [0])
  filter_length = self.filter_length
  _0 = torch.add(torch.div(filter_length, 2), 1)
  cutoff = int(_0)
  _1 = torch.slice(torch.slice(forward_transform), 1, None, cutoff)
  real_part = torch.to(torch.slice(_1, 2), 6)
  _2 = torch.slice(torch.slice(forward_transform), 1, cutoff)
  imag_part = torch.to(torch.slice(_2, 2), 6)
  _3 = torch.add(torch.pow(real_part, 2), torch.pow(imag_part, 2))
  magnitude = torch.sqrt(_3)
  phase = torch.atan2(ops.prim.data(imag_part), ops.prim.data(real_part))
  return (magnitude, phase)
```
And the gives us enough information to start writing a python model mimicking
the original model.

I also used a profile logger while running the [C++ example](https://github.com/snakers4/silero-vad/tree/master/examples/cpp)
in the silero-vad repository to get more information while running it using
onnx profiler.

I made the following changes to get it to compile and add the profiling:
```console
$ git diff
diff --git a/examples/cpp/silero-vad-onnx.cpp b/examples/cpp/silero-vad-onnx.cpp
index 380d76d..99fb289 100644
--- a/examples/cpp/silero-vad-onnx.cpp
+++ b/examples/cpp/silero-vad-onnx.cpp
@@ -131,7 +131,7 @@ private:
     timestamp_t current_speech;

     // Loads the ONNX model.
-    void init_onnx_model(const std::wstring& model_path) {
+    void init_onnx_model(const std::string& model_path) {
         init_engine_threads(1, 1);
         session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
     }
@@ -303,7 +303,7 @@ public:
 public:
     // Constructor: sets model path, sample rate, window size (ms), and other parameters.
     // The parameters are set to match the Python version.
-    VadIterator(const std::wstring ModelPath,
+    VadIterator(const std::string ModelPath,
         int Sample_rate = 16000, int windows_frame_size = 32,
         float Threshold = 0.5, int min_silence_duration_ms = 100,
         int speech_pad_ms = 30, int min_speech_duration_ms = 250,
@@ -329,7 +329,7 @@ public:

 int main() {
     // Read the WAV file (expects 16000 Hz, mono, PCM).
-    wav::WavReader wav_reader("audio/recorder.wav"); // File located in the "audio" folder.
+    wav::WavReader wav_reader("jfk.wav"); // File located in the "audio" folder.
     int numSamples = wav_reader.num_samples();
     std::vector<float> input_wav(static_cast<size_t>(numSamples));
     for (size_t i = 0; i < static_cast<size_t>(numSamples); i++) {
@@ -337,7 +337,7 @@ int main() {
     }

     // Set the ONNX model path (file located in the "model" folder).
-    std::wstring model_path = L"model/silero_vad.onnx";
+    std::string model_path = "../../src/silero_vad/data/silero_vad.onnx";

     // Initialize the VadIterator.
     VadIterator vad(model_path);
```
And a make file to build and run.
```console
ONNX_PATH=./onnxruntime-linux-x64-1.12.1

build:
	g++ silero-vad-onnx.cpp -I ${ONNX_PATH}/include/ \
		-L ${ONNX_PATH}/lib/ \
		-lonnxruntime \
		-Wl,-rpath,${ONNX_PATH}/lib/ \
		-o test

run: build
	./test
```
Now, this will create files in the current directory and if we inspect it we
can find some information about the shapes and operations, again for example
looking at the STFT layer:
```
20 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_before","args" : {"op_name" : "Pad"}},
   21 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "32","activation_size" : "2304","output_t      ype_shape" : [{"float":[1,640]}],"exec_plan_index" : "7","graph_index" : "7","input_type_shape" : [{"float":[1,576]},{"int64":[4]}],"provider" : "CPUExecutionProvider","op_name" : "Pad"}},
   22 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_after","args" : {"op_name" : "Pad"}},
   23 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_before","args" : {"op_name" : "Unsqueeze"}},
   24 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59313,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "8","activation_size" : "2560","output_type      _shape" : [{"float":[1,1,640]}],"exec_plan_index" : "8","graph_index" : "8","input_type_shape" : [{"float":[1,640]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Unsqueeze"}},
   25 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59317,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_after","args" : {"op_name" : "Unsqueeze"}},
   26 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_before","args" : {"op_name" : "Conv"}},
   27 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :54,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "4128","parameter_size" : "264192","activation_size" : "2560","output_typ      e_shape" : [{"float":[1,258,4]}],"exec_plan_index" : "9","graph_index" : "9","input_type_shape" : [{"float":[1,1,640]},{"float":[258,1,256]}],"provider" : "CPUExecutionProvider","op_name" : "Conv"}},
   28 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59375,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_after","args" : {"op_name" : "Conv"}},
   29 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59376,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_before","args" : {"op_name" : "Slice"}},
   30 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59377,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "14","graph_index" : "14","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   31 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59385,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_after","args" : {"op_name" : "Slice"}},
   32 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_before","args" : {"op_name" : "Pow"}},
   33 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_sha      pe" : [{"float":[1,129,4]}],"exec_plan_index" : "18","graph_index" : "18","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   34 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59391,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_after","args" : {"op_name" : "Pow"}},
   35 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_before","args" : {"op_name" : "Slice"}},
   36 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "11","graph_index" : "11","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   37 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59398,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_after","args" : {"op_name" : "Slice"}},
   38 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_before","args" : {"op_name" : "Pow"}},
   39 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "17","graph_index" : "17","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   40 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59403,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_after","args" : {"op_name" : "Pow"}},
   41 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_before","args" : {"op_name" : "Add"}},
   42 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "4128","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "19","graph_index" : "19","input_type_shape" : [{"float":[1,129,4]},{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Add"}},
   43 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59409,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_after","args" : {"op_name" : "Add"}},
   44 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59410,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_fence_before","args" : {"op_name" : "Sqrt"}},
   45 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59411,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "2064","output_type_shap      e" : [{"float":[1,129,4]}],"exec_plan_index" : "20","graph_index" : "20","input_type_shape" : [{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Sqrt"}},
```
So with that I created a script to extract the tensors from the JIT model, and
then implement the model in python using torch (this was not as straightforward
as it might sound here). The converted model can be found in
[audio/silero-vad](../audio/silero-vad/src/reverse-eng/).

Using the `jfk.wav` with the original silero-vad model these are the predictions:
```console
$ python src/jfk.py
[0] probability: 0.0120120458
[1] probability: 0.0106779542
[2] probability: 0.1321811974
[3] probability: 0.0654894710
[4] probability: 0.0445981026
[5] probability: 0.0223348271
[6] probability: 0.0260702968
[7] probability: 0.0116709163
[8] probability: 0.0081158215
[9] probability: 0.0067158826
[10] probability: 0.8111256361
[11] probability: 0.9633629322
[12] probability: 0.9310814142
[13] probability: 0.7854600549
[14] probability: 0.8146636486
[15] probability: 0.9672259092
[16] probability: 0.9664794803
[17] probability: 0.9530465603
[18] probability: 0.9773806334
[19] probability: 0.9515406489
[20] probability: 0.9235361218
[21] probability: 0.9561933875
[22] probability: 0.9820070863
[23] probability: 0.9649533629
[24] probability: 0.9486407042
[25] probability: 0.9199727774
[26] probability: 0.8507736921
[27] probability: 0.8281027675
[28] probability: 0.7944886088
[29] probability: 0.8232654929
[30] probability: 0.8295858502
[31] probability: 0.8030185699
[32] probability: 0.9451498985
[33] probability: 0.8949782848
[34] probability: 0.9032012224
[35] probability: 0.9025285244
[36] probability: 0.9016729593
[37] probability: 0.9391839504
[38] probability: 0.9761081338
[39] probability: 0.9765046835
[40] probability: 0.9614208937
[41] probability: 0.8868988156
[42] probability: 0.9054323435
[43] probability: 0.9729118943
[44] probability: 0.9419037700
[45] probability: 0.9301297665
[46] probability: 0.9049455523
[47] probability: 0.9192379713
[48] probability: 0.9463497400
[49] probability: 0.8815279603
[50] probability: 0.8565585017
[51] probability: 0.8562414050
[52] probability: 0.9654588103
[53] probability: 0.9728733301
[54] probability: 0.9644294381
[55] probability: 0.9485635757
[56] probability: 0.9216982722
[57] probability: 0.9331229329
[58] probability: 0.9122018218
[59] probability: 0.8967185020
[60] probability: 0.8226366639
[61] probability: 0.9199765325
[62] probability: 0.9400870204
[63] probability: 0.9173651338
[64] probability: 0.7509807944
[65] probability: 0.9055827856
[66] probability: 0.7457287908
[67] probability: 0.5512428880
[68] probability: 0.3979139328
[69] probability: 0.1968278140
[70] probability: 0.1336449385
[71] probability: 0.0854208618
[72] probability: 0.0512541421
[73] probability: 0.0213614609
[74] probability: 0.0087905023
[75] probability: 0.0050795670
[76] probability: 0.0036360626
[77] probability: 0.0029130245
[78] probability: 0.0062774355
[79] probability: 0.0060164286
[80] probability: 0.0036439903
[81] probability: 0.0020915263
[82] probability: 0.0016147600
[83] probability: 0.0014228907
[84] probability: 0.0012152086
[85] probability: 0.0016399137
[86] probability: 0.0007782626
[87] probability: 0.0012633284
[88] probability: 0.0011245427
[89] probability: 0.0012091363
[90] probability: 0.0024176650
[91] probability: 0.0012052481
[92] probability: 0.0006706595
[93] probability: 0.0004096399
[94] probability: 0.0005879427
[95] probability: 0.0003524663
[96] probability: 0.0006112840
[97] probability: 0.0004331781
[98] probability: 0.0012036999
[99] probability: 0.0004305345
[100] probability: 0.0003728746
[101] probability: 0.0005225370
[102] probability: 0.0181077644
[103] probability: 0.2840053737
[104] probability: 0.4943751991
[105] probability: 0.6032722592
[106] probability: 0.5436841846
[107] probability: 0.5589256883
[108] probability: 0.5062690377
[109] probability: 0.3489521742
[110] probability: 0.2822247148
[111] probability: 0.2502350807
[112] probability: 0.4467811286
[113] probability: 0.8259999156
[114] probability: 0.7331997752
[115] probability: 0.6296291947
[116] probability: 0.3672357500
[117] probability: 0.2370462567
[118] probability: 0.1266799271
[119] probability: 0.0863809213
[120] probability: 0.0577458292
[121] probability: 0.0126561048
[122] probability: 0.0058960663
[123] probability: 0.0066269282
[124] probability: 0.1001239195
[125] probability: 0.2267967612
[126] probability: 0.7768144608
[127] probability: 0.8881686330
[128] probability: 0.8496580720
[129] probability: 0.8128281832
[130] probability: 0.7959909439
[131] probability: 0.7936805487
[132] probability: 0.6849995852
[133] probability: 0.6143192053
[134] probability: 0.6148759723
[135] probability: 0.3155294359
[136] probability: 0.1442092806
[137] probability: 0.0466341823
[138] probability: 0.0278023500
[139] probability: 0.0182217564
[140] probability: 0.0088233612
[141] probability: 0.0095065329
[142] probability: 0.0044769812
[143] probability: 0.0031184589
[144] probability: 0.0016689267
[145] probability: 0.0023460335
[146] probability: 0.0007922960
[147] probability: 0.0028725227
[148] probability: 0.0011672110
[149] probability: 0.0020256110
[150] probability: 0.0020782938
[151] probability: 0.0009769580
[152] probability: 0.0009220199
[153] probability: 0.0024484431
[154] probability: 0.0046779355
[155] probability: 0.0024497141
[156] probability: 0.0018141053
[157] probability: 0.0012290307
[158] probability: 0.0009697533
[159] probability: 0.0011016321
[160] probability: 0.0010593801
[161] probability: 0.0018238472
[162] probability: 0.0009759203
[163] probability: 0.0007066324
[164] probability: 0.0006456191
[165] probability: 0.0013584567
[166] probability: 0.0006583764
[167] probability: 0.0014947796
[168] probability: 0.0012043880
[169] probability: 0.7265451550
[170] probability: 0.8275423646
[171] probability: 0.7977938652
[172] probability: 0.8630424142
[173] probability: 0.8654760718
[174] probability: 0.8103245497
[175] probability: 0.8888602853
[176] probability: 0.8212413788
[177] probability: 0.8759981394
[178] probability: 0.8938365579
[179] probability: 0.9498395920
[180] probability: 0.9528379440
[181] probability: 0.9281737804
[182] probability: 0.9655471444
[183] probability: 0.9308375120
[184] probability: 0.7055628300
[185] probability: 0.7111269236
[186] probability: 0.7741216421
[187] probability: 0.9365538359
[188] probability: 0.9579713345
[189] probability: 0.9543846846
[190] probability: 0.9655930400
[191] probability: 0.9719272852
[192] probability: 0.9262279868
[193] probability: 0.9058678150
[194] probability: 0.9641671181
[195] probability: 0.9669665694
[196] probability: 0.9642478228
[197] probability: 0.9578868747
[198] probability: 0.9696291685
[199] probability: 0.9675853848
[200] probability: 0.9736673832
[201] probability: 0.9782630801
[202] probability: 0.9718350172
[203] probability: 0.9790894389
[204] probability: 0.9796285033
[205] probability: 0.9683827758
[206] probability: 0.9775854349
[207] probability: 0.9806787968
[208] probability: 0.9728345275
[209] probability: 0.9756219387
[210] probability: 0.9809038043
[211] probability: 0.9838793874
[212] probability: 0.9794865847
[213] probability: 0.9755236506
[214] probability: 0.9861546159
[215] probability: 0.9860898256
[216] probability: 0.9652890563
[217] probability: 0.9375467896
[218] probability: 0.8729331493
[219] probability: 0.9602597356
[220] probability: 0.9770282507
[221] probability: 0.9740325809
[222] probability: 0.9711015224
[223] probability: 0.9773187041
[224] probability: 0.9826788902
[225] probability: 0.9807331562
[226] probability: 0.9793297648
[227] probability: 0.9699174166
[228] probability: 0.9644414783
[229] probability: 0.9687068462
[230] probability: 0.9681332111
[231] probability: 0.9577034712
[232] probability: 0.9728902578
[233] probability: 0.9768581390
[234] probability: 0.9660829902
[235] probability: 0.9392179251
[236] probability: 0.8526392579
[237] probability: 0.5641263723
[238] probability: 0.2013247162
[239] probability: 0.1066527665
[240] probability: 0.0310583655
[241] probability: 0.0157215856
[242] probability: 0.0130914506
[243] probability: 0.0044830763
[244] probability: 0.0069351792
[245] probability: 0.0024922192
[246] probability: 0.0027061312
[247] probability: 0.0020019219
[248] probability: 0.0011804849
[249] probability: 0.0020268953
[250] probability: 0.0007212795
[251] probability: 0.0010822940
[252] probability: 0.0006232891
[253] probability: 0.0007668757
[254] probability: 0.0010701693
[255] probability: 0.0008318963
[256] probability: 0.6040052772
[257] probability: 0.8263453245
[258] probability: 0.7968529463
[259] probability: 0.7766568065
[260] probability: 0.7201390862
[261] probability: 0.6486166120
[262] probability: 0.6912519336
[263] probability: 0.8114249706
[264] probability: 0.9494416118
[265] probability: 0.9156619310
[266] probability: 0.8544524908
[267] probability: 0.5401257277
[268] probability: 0.2489332706
[269] probability: 0.2530018091
[270] probability: 0.9610205293
[271] probability: 0.9751768708
[272] probability: 0.9762893319
[273] probability: 0.9723498225
[274] probability: 0.9672072530
[275] probability: 0.9852018952
[276] probability: 0.9831889272
[277] probability: 0.9588776827
[278] probability: 0.9845831394
[279] probability: 0.9928609729
[280] probability: 0.9928285480
[281] probability: 0.9938415289
[282] probability: 0.9930645823
[283] probability: 0.9940449595
[284] probability: 0.9893879890
[285] probability: 0.9955287576
[286] probability: 0.9922802448
[287] probability: 0.9916920662
[288] probability: 0.9911763668
[289] probability: 0.9976255298
[290] probability: 0.9954389930
[291] probability: 0.9819942117
[292] probability: 0.9935252666
[293] probability: 0.9948412776
[294] probability: 0.9961053729
[295] probability: 0.9935407043
[296] probability: 0.9927965999
[297] probability: 0.9935270548
[298] probability: 0.9914484620
[299] probability: 0.9969546795
[300] probability: 0.9937365055
[301] probability: 0.9903761744
[302] probability: 0.9891674519
[303] probability: 0.9745979309
[304] probability: 0.9902220964
[305] probability: 0.9918566942
[306] probability: 0.9890555143
[307] probability: 0.9960351586
[308] probability: 0.9963703156
[309] probability: 0.9965223074
[310] probability: 0.9939389229
[311] probability: 0.9927075505
[312] probability: 0.9939052463
[313] probability: 0.9901870489
[314] probability: 0.9839034081
[315] probability: 0.9867933393
[316] probability: 0.9954883456
[317] probability: 0.9951952100
[318] probability: 0.9929647446
[319] probability: 0.9927020073
[320] probability: 0.9865319133
[321] probability: 0.9708641768
[322] probability: 0.9639129043
[323] probability: 0.9845443368
[324] probability: 0.9337452650
[325] probability: 0.9636278749
[326] probability: 0.9665008783
[327] probability: 0.9497909546
[328] probability: 0.7639142275
[329] probability: 0.4962018728
[330] probability: 0.4612325430
[331] probability: 0.0821653679
[332] probability: 0.0405023694
[333] probability: 0.0188933071
[334] probability: 0.0258657020
[335] probability: 0.0101035936
[336] probability: 0.0146565679
[337] probability: 0.0091484794
[338] probability: 0.0068754503
[339] probability: 0.0583271906
[340] probability: 0.0139130643
[341] probability: 0.0289103184
[342] probability: 0.0327798538
[343] probability: 0.0748589560
```
And these are the predictions using the converted model:
```console
$ python src/reverse-eng/test_conv_model.py
Loading PyTorch model from silero_vad_conv_pytorch.pth
Original wav shape: torch.Size([176000]), duration: 11.00 seconds
Processing 344 chunks of 512 samples each
Initializing state
Processed chunk 1/344, probability: 0.012253
Processed chunk 2/344, probability: 0.007223
Processed chunk 3/344, probability: 0.366389
Processed chunk 4/344, probability: 0.089219
Processed chunk 5/344, probability: 0.029518
Processed chunk 6/344, probability: 0.014793
Processed chunk 7/344, probability: 0.013684
Processed chunk 8/344, probability: 0.013345
Processed chunk 9/344, probability: 0.010185
Processed chunk 10/344, probability: 0.014257
Processed chunk 11/344, probability: 0.245753
Processed chunk 12/344, probability: 0.057528
Processed chunk 13/344, probability: 0.179109
Processed chunk 14/344, probability: 0.338114
Processed chunk 15/344, probability: 0.147751
Processed chunk 16/344, probability: 0.932849
Processed chunk 17/344, probability: 0.964219
Processed chunk 18/344, probability: 0.954369
Processed chunk 19/344, probability: 0.960660
Processed chunk 20/344, probability: 0.927135
Processed chunk 21/344, probability: 0.842062
Processed chunk 22/344, probability: 0.100232
Processed chunk 23/344, probability: 0.564500
Processed chunk 24/344, probability: 0.741666
Processed chunk 25/344, probability: 0.661513
Processed chunk 26/344, probability: 0.571926
Processed chunk 27/344, probability: 0.381262
Processed chunk 28/344, probability: 0.468206
Processed chunk 29/344, probability: 0.566361
Processed chunk 30/344, probability: 0.468684
Processed chunk 31/344, probability: 0.891847
Processed chunk 32/344, probability: 0.871961
Processed chunk 33/344, probability: 0.415601
Processed chunk 34/344, probability: 0.402962
Processed chunk 35/344, probability: 0.479159
Processed chunk 36/344, probability: 0.679892
Processed chunk 37/344, probability: 0.141078
Processed chunk 38/344, probability: 0.817669
Processed chunk 39/344, probability: 0.436826
Processed chunk 40/344, probability: 0.960370
Processed chunk 41/344, probability: 0.974859
Processed chunk 42/344, probability: 0.961204
Processed chunk 43/344, probability: 0.954828
Processed chunk 44/344, probability: 0.309893
Processed chunk 45/344, probability: 0.539230
Processed chunk 46/344, probability: 0.757563
Processed chunk 47/344, probability: 0.673551
Processed chunk 48/344, probability: 0.656830
Processed chunk 49/344, probability: 0.562993
Processed chunk 50/344, probability: 0.621347
Processed chunk 51/344, probability: 0.851591
Processed chunk 52/344, probability: 0.623764
Processed chunk 53/344, probability: 0.761936
Processed chunk 54/344, probability: 0.954612
Processed chunk 55/344, probability: 0.166623
Processed chunk 56/344, probability: 0.868001
Processed chunk 57/344, probability: 0.793257
Processed chunk 58/344, probability: 0.777915
Processed chunk 59/344, probability: 0.770807
Processed chunk 60/344, probability: 0.677196
Processed chunk 61/344, probability: 0.875169
Processed chunk 62/344, probability: 0.668913
Processed chunk 63/344, probability: 0.923832
Processed chunk 64/344, probability: 0.952841
Processed chunk 65/344, probability: 0.869377
Processed chunk 66/344, probability: 0.950902
Processed chunk 67/344, probability: 0.898180
Processed chunk 68/344, probability: 0.896478
Processed chunk 69/344, probability: 0.831571
Processed chunk 70/344, probability: 0.759298
Processed chunk 71/344, probability: 0.739200
Processed chunk 72/344, probability: 0.683064
Processed chunk 73/344, probability: 0.501261
Processed chunk 74/344, probability: 0.322918
Processed chunk 75/344, probability: 0.196016
Processed chunk 76/344, probability: 0.093478
Processed chunk 77/344, probability: 0.248194
Processed chunk 78/344, probability: 0.123136
Processed chunk 79/344, probability: 0.129942
Processed chunk 80/344, probability: 0.107908
Processed chunk 81/344, probability: 0.072728
Processed chunk 82/344, probability: 0.059153
Processed chunk 83/344, probability: 0.048282
Processed chunk 84/344, probability: 0.027020
Processed chunk 85/344, probability: 0.019829
Processed chunk 86/344, probability: 0.017090
Processed chunk 87/344, probability: 0.011653
Processed chunk 88/344, probability: 0.012197
Processed chunk 89/344, probability: 0.012900
Processed chunk 90/344, probability: 0.018752
Processed chunk 91/344, probability: 0.010503
Processed chunk 92/344, probability: 0.008380
Processed chunk 93/344, probability: 0.008858
Processed chunk 94/344, probability: 0.031318
Processed chunk 95/344, probability: 0.012537
Processed chunk 96/344, probability: 0.011189
Processed chunk 97/344, probability: 0.008230
Processed chunk 98/344, probability: 0.008199
Processed chunk 99/344, probability: 0.007536
Processed chunk 100/344, probability: 0.006285
Processed chunk 101/344, probability: 0.005267
Processed chunk 102/344, probability: 0.007974
Processed chunk 103/344, probability: 0.026743
Processed chunk 104/344, probability: 0.334263
Processed chunk 105/344, probability: 0.705672
Processed chunk 106/344, probability: 0.707406
Processed chunk 107/344, probability: 0.740729
Processed chunk 108/344, probability: 0.504565
Processed chunk 109/344, probability: 0.520441
Processed chunk 110/344, probability: 0.787809
Processed chunk 111/344, probability: 0.794517
Processed chunk 112/344, probability: 0.411491
Processed chunk 113/344, probability: 0.634105
Processed chunk 114/344, probability: 0.824849
Processed chunk 115/344, probability: 0.967505
Processed chunk 116/344, probability: 0.979102
Processed chunk 117/344, probability: 0.880362
Processed chunk 118/344, probability: 0.613459
Processed chunk 119/344, probability: 0.519005
Processed chunk 120/344, probability: 0.355173
Processed chunk 121/344, probability: 0.227172
Processed chunk 122/344, probability: 0.093588
Processed chunk 123/344, probability: 0.051443
Processed chunk 124/344, probability: 0.036004
Processed chunk 125/344, probability: 0.138002
Processed chunk 126/344, probability: 0.121147
Processed chunk 127/344, probability: 0.280491
Processed chunk 128/344, probability: 0.820747
Processed chunk 129/344, probability: 0.636644
Processed chunk 130/344, probability: 0.806154
Processed chunk 131/344, probability: 0.912038
Processed chunk 132/344, probability: 0.804077
Processed chunk 133/344, probability: 0.678623
Processed chunk 134/344, probability: 0.414706
Processed chunk 135/344, probability: 0.822079
Processed chunk 136/344, probability: 0.712333
Processed chunk 137/344, probability: 0.504102
Processed chunk 138/344, probability: 0.307678
Processed chunk 139/344, probability: 0.131040
Processed chunk 140/344, probability: 0.054814
Processed chunk 141/344, probability: 0.031848
Processed chunk 142/344, probability: 0.027266
Processed chunk 143/344, probability: 0.043220
Processed chunk 144/344, probability: 0.033395
Processed chunk 145/344, probability: 0.040276
Processed chunk 146/344, probability: 0.038370
Processed chunk 147/344, probability: 0.017125
Processed chunk 148/344, probability: 0.020428
Processed chunk 149/344, probability: 0.008985
Processed chunk 150/344, probability: 0.022063
Processed chunk 151/344, probability: 0.005423
Processed chunk 152/344, probability: 0.005655
Processed chunk 153/344, probability: 0.006187
Processed chunk 154/344, probability: 0.012502
Processed chunk 155/344, probability: 0.006069
Processed chunk 156/344, probability: 0.022263
Processed chunk 157/344, probability: 0.010274
Processed chunk 158/344, probability: 0.012517
Processed chunk 159/344, probability: 0.004559
Processed chunk 160/344, probability: 0.008382
Processed chunk 161/344, probability: 0.004786
Processed chunk 162/344, probability: 0.003169
Processed chunk 163/344, probability: 0.006468
Processed chunk 164/344, probability: 0.004081
Processed chunk 165/344, probability: 0.003139
Processed chunk 166/344, probability: 0.004829
Processed chunk 167/344, probability: 0.006377
Processed chunk 168/344, probability: 0.002778
Processed chunk 169/344, probability: 0.004988
Processed chunk 170/344, probability: 0.070760
Processed chunk 171/344, probability: 0.011600
Processed chunk 172/344, probability: 0.073784
Processed chunk 173/344, probability: 0.058416
Processed chunk 174/344, probability: 0.298270
Processed chunk 175/344, probability: 0.692843
Processed chunk 176/344, probability: 0.865757
Processed chunk 177/344, probability: 0.764916
Processed chunk 178/344, probability: 0.952728
Processed chunk 179/344, probability: 0.970718
Processed chunk 180/344, probability: 0.847768
Processed chunk 181/344, probability: 0.924314
Processed chunk 182/344, probability: 0.763485
Processed chunk 183/344, probability: 0.795622
Processed chunk 184/344, probability: 0.959421
Processed chunk 185/344, probability: 0.753841
Processed chunk 186/344, probability: 0.650826
Processed chunk 187/344, probability: 0.787271
Processed chunk 188/344, probability: 0.904657
Processed chunk 189/344, probability: 0.438222
Processed chunk 190/344, probability: 0.447936
Processed chunk 191/344, probability: 0.920199
Processed chunk 192/344, probability: 0.922720
Processed chunk 193/344, probability: 0.798727
Processed chunk 194/344, probability: 0.684547
Processed chunk 195/344, probability: 0.442535
Processed chunk 196/344, probability: 0.948953
Processed chunk 197/344, probability: 0.955180
Processed chunk 198/344, probability: 0.977817
Processed chunk 199/344, probability: 0.989137
Processed chunk 200/344, probability: 0.991587
Processed chunk 201/344, probability: 0.992916
Processed chunk 202/344, probability: 0.991625
Processed chunk 203/344, probability: 0.988809
Processed chunk 204/344, probability: 0.465130
Processed chunk 205/344, probability: 0.959194
Processed chunk 206/344, probability: 0.952062
Processed chunk 207/344, probability: 0.957344
Processed chunk 208/344, probability: 0.975708
Processed chunk 209/344, probability: 0.949765
Processed chunk 210/344, probability: 0.975393
Processed chunk 211/344, probability: 0.980040
Processed chunk 212/344, probability: 0.989174
Processed chunk 213/344, probability: 0.994864
Processed chunk 214/344, probability: 0.994398
Processed chunk 215/344, probability: 0.995075
Processed chunk 216/344, probability: 0.995268
Processed chunk 217/344, probability: 0.995031
Processed chunk 218/344, probability: 0.992572
Processed chunk 219/344, probability: 0.969323
Processed chunk 220/344, probability: 0.841870
Processed chunk 221/344, probability: 0.990073
Processed chunk 222/344, probability: 0.992081
Processed chunk 223/344, probability: 0.992284
Processed chunk 224/344, probability: 0.996371
Processed chunk 225/344, probability: 0.983920
Processed chunk 226/344, probability: 0.979731
Processed chunk 227/344, probability: 0.994697
Processed chunk 228/344, probability: 0.995321
Processed chunk 229/344, probability: 0.996780
Processed chunk 230/344, probability: 0.996463
Processed chunk 231/344, probability: 0.996155
Processed chunk 232/344, probability: 0.953821
Processed chunk 233/344, probability: 0.977058
Processed chunk 234/344, probability: 0.956587
Processed chunk 235/344, probability: 0.979999
Processed chunk 236/344, probability: 0.966082
Processed chunk 237/344, probability: 0.946846
Processed chunk 238/344, probability: 0.914940
Processed chunk 239/344, probability: 0.873253
Processed chunk 240/344, probability: 0.677977
Processed chunk 241/344, probability: 0.535046
Processed chunk 242/344, probability: 0.427438
Processed chunk 243/344, probability: 0.337373
Processed chunk 244/344, probability: 0.287535
Processed chunk 245/344, probability: 0.299001
Processed chunk 246/344, probability: 0.292553
Processed chunk 247/344, probability: 0.211759
Processed chunk 248/344, probability: 0.099344
Processed chunk 249/344, probability: 0.057190
Processed chunk 250/344, probability: 0.059714
Processed chunk 251/344, probability: 0.022881
Processed chunk 252/344, probability: 0.018783
Processed chunk 253/344, probability: 0.020064
Processed chunk 254/344, probability: 0.015387
Processed chunk 255/344, probability: 0.006533
Processed chunk 256/344, probability: 0.005338
Processed chunk 257/344, probability: 0.880338
Processed chunk 258/344, probability: 0.025980
Processed chunk 259/344, probability: 0.533862
Processed chunk 260/344, probability: 0.705605
Processed chunk 261/344, probability: 0.542060
Processed chunk 262/344, probability: 0.608190
Processed chunk 263/344, probability: 0.683656
Processed chunk 264/344, probability: 0.564003
Processed chunk 265/344, probability: 0.827814
Processed chunk 266/344, probability: 0.935665
Processed chunk 267/344, probability: 0.937294
Processed chunk 268/344, probability: 0.787430
Processed chunk 269/344, probability: 0.562007
Processed chunk 270/344, probability: 0.430153
Processed chunk 271/344, probability: 0.163181
Processed chunk 272/344, probability: 0.615059
Processed chunk 273/344, probability: 0.684973
Processed chunk 274/344, probability: 0.904332
Processed chunk 275/344, probability: 0.971613
Processed chunk 276/344, probability: 0.970967
Processed chunk 277/344, probability: 0.975359
Processed chunk 278/344, probability: 0.966694
Processed chunk 279/344, probability: 0.945710
Processed chunk 280/344, probability: 0.969530
Processed chunk 281/344, probability: 0.976718
Processed chunk 282/344, probability: 0.985269
Processed chunk 283/344, probability: 0.989273
Processed chunk 284/344, probability: 0.986473
Processed chunk 285/344, probability: 0.988614
Processed chunk 286/344, probability: 0.986563
Processed chunk 287/344, probability: 0.989617
Processed chunk 288/344, probability: 0.990141
Processed chunk 289/344, probability: 0.978997
Processed chunk 290/344, probability: 0.982440
Processed chunk 291/344, probability: 0.986247
Processed chunk 292/344, probability: 0.962913
Processed chunk 293/344, probability: 0.994208
Processed chunk 294/344, probability: 0.993430
Processed chunk 295/344, probability: 0.990184
Processed chunk 296/344, probability: 0.979956
Processed chunk 297/344, probability: 0.996237
Processed chunk 298/344, probability: 0.992983
Processed chunk 299/344, probability: 0.985990
Processed chunk 300/344, probability: 0.991898
Processed chunk 301/344, probability: 0.995755
Processed chunk 302/344, probability: 0.995893
Processed chunk 303/344, probability: 0.995428
Processed chunk 304/344, probability: 0.995688
Processed chunk 305/344, probability: 0.998187
Processed chunk 306/344, probability: 0.996501
Processed chunk 307/344, probability: 0.998080
Processed chunk 308/344, probability: 0.998410
Processed chunk 309/344, probability: 0.997358
Processed chunk 310/344, probability: 0.996707
Processed chunk 311/344, probability: 0.998803
Processed chunk 312/344, probability: 0.996651
Processed chunk 313/344, probability: 0.996675
Processed chunk 314/344, probability: 0.996512
Processed chunk 315/344, probability: 0.996138
Processed chunk 316/344, probability: 0.731368
Processed chunk 317/344, probability: 0.969654
Processed chunk 318/344, probability: 0.983222
Processed chunk 319/344, probability: 0.990783
Processed chunk 320/344, probability: 0.985080
Processed chunk 321/344, probability: 0.985515
Processed chunk 322/344, probability: 0.977458
Processed chunk 323/344, probability: 0.929311
Processed chunk 324/344, probability: 0.993227
Processed chunk 325/344, probability: 0.968551
Processed chunk 326/344, probability: 0.979412
Processed chunk 327/344, probability: 0.981892
Processed chunk 328/344, probability: 0.977588
Processed chunk 329/344, probability: 0.934229
Processed chunk 330/344, probability: 0.894397
Processed chunk 331/344, probability: 0.047900
Processed chunk 332/344, probability: 0.294567
Processed chunk 333/344, probability: 0.364360
Processed chunk 334/344, probability: 0.335913
Processed chunk 335/344, probability: 0.512133
Processed chunk 336/344, probability: 0.839461
Processed chunk 337/344, probability: 0.711789
Processed chunk 338/344, probability: 0.469788
Processed chunk 339/344, probability: 0.396873
Processed chunk 340/344, probability: 0.449585
Processed chunk 341/344, probability: 0.387324
Processed chunk 342/344, probability: 0.621481
Processed chunk 343/344, probability: 0.541268
Processed chunk 344/344, probability: 0.792273

Processing complete!
Processed 344 chunks of audio
Average probability: 0.581019
Max probability: 0.998803
Min probability: 0.002778
```
My hope if that these are close enough to be of use for the whiper.cpp
implementation.


The inputs to the model are the following, which I collected from https://netron.app/
using the ONNX model:
```
Name   
input: tensor: float32[?,?]

state: tensor: float32[2,?,128]

sr   : tensor: int64    (sampling rate)
```
This is what the model looks like:

![image](./images/silero_vad.onnx.png)

The first node is checking that the sample rate is 16000. The model also
supports 8000Hz so this is probably what this check is about.

In the actual processing there first node is the STFT (Short-Time Fourier
Transform):
```
Padding operation:
Input:  [1, 576]        // 512 sample plus 64 samples from previous frame
Output: [1, 640]        // 567 + 64 = 640

Reshape/squeeze operation:
Reshapes [1,640] to [1,1,640] (adds dimension for batch)

conv1d operation:
Input shape: [1,1,640] (batch, channels, time)
Filter shape: [258,1,256] (num_filters, in_channels, kernel_size)
Output shape: [1,258,4]
Parameters: 264,192 (258 filters × 1 channel × 256 kernel + biases)

slice operation:
Slicing to [1,129,4] suggests extracting real components from complex numbers
```
This is then followed by a 4 encoder layers:
```
First encoder layer:
conv1d operation:
Input shape: [1,129,4] (batch, channels, time)
Filter shape: [128,129,3] (num_filters, in_channels, kernel_size)
Output shape: [1,128,4]
Parameters: 198,656 (128 filters × 129 channels × 3 kernel + biases)

Second encoder layer:
conv1d operation:
Input shape: [1,128,4] (batch, channels, time)
Filter shape: [64,128,3] (num_filters, in_channels, kernel_size)
Output shape: [1,64,2]
Parameters: 98,560 (64 filters × 128 channels × 3 kernel + biases)

Third encoder layer:
conv1d operation:
Input shape: [1,64,2] (batch, channels, time)
Filter shape: [64,64,3] (num_filters, in_channels, kernel_size)
Output shape: [1,64,1] 
Parameters: 49,408 (64 filters × 64 channels × 3 kernel + biases)

Fourth encoder layer:
conv1d operation:
Input shape: [1,64,1] (batch, channels, time)
Filter shape: [128,64,3] (num_filters, in_channels, kernel_size)
Output shape: [1,128,1]
Parameters: 98,816 (128 filters × 64 channels × 3 kernel + biases)
```

Decoder with LSTM:
```
sequeeze operation:
Removing dimension: [1,128,1] → [1,128]

LSTM operation:
- Input: [1,1,128] (sequence of 1 with 128 features)
- Parameters: 1024 weights
- Outputs: [1,1,1,128], [1,1,128], [1,1,128] (output, hidden, cell states)

ReLU operation:

Conv1d operation:
Input shape: [1,128,1] (batch, channels, time)
Filter shape: [1,128,1] (num_filters, in_channels, kernel_size)
Output shape: [1,1,1]
Parameters: 516 (1 filter × 128 channels × 1 kernel + biases)
```

There are two main components in the model, one named `_model` which takes care
of the 16kHz audio signal and the other one is named `_model_8k` which takes
care of the 8kHz audio signal. 

Both have the same layers but there tensor shapes might be different (more on
this later when we look at them).

### Tensors
The following are the tensor that are in the the model (only focusing on the 16kHz
and skipping the 8kHz model for now):
```console
Tensors to be written:
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
_model.encoder.0.reparam_conv.weight: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias: torch.Size([128])
_model.encoder.1.reparam_conv.weight: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias: torch.Size([64])
_model.encoder.2.reparam_conv.weight: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias: torch.Size([64])
_model.encoder.3.reparam_conv.weight: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias: torch.Size([128])
_model.decoder.rnn.weight_ih: torch.Size([512, 128])
_model.decoder.rnn.weight_hh: torch.Size([512, 128])
_model.decoder.rnn.bias_ih: torch.Size([512])
_model.decoder.rnn.bias_hh: torch.Size([512])
_model.decoder.decoder.2.weight: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias: torch.Size([1])
```

#### Short-Time Fourier Transform (STFT)
So if we start with an raw audio input signal, this will first be sampled and
quantized, which will give us a vector of floats.

Next we divide this into frames/segments of the samples that usually overlap to
avoid spectral leakage, and the size of a frame is usually a power of two so
that we can use the Fast Fourier Transform. 
If we look closely that the node above we find this:
```
    (stft): RecursiveScriptModule(
      original_name=STFT
      (padding): RecursiveScriptModule(original_name=ReflectionPad1d)
    )
```
I missed this initially but this is an `ReflectionPad1d` which is not a simple
zero padding operation. This will add padding to the left and right of the
samples, and will use values from the respective sides.

If we inspect the models tensors (see below for details) we find that the
model contains a precomputed STFT basis buffer:
```console
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
```
The first dimension is the filter length, which in this case is 258 and probably
129 complex number (real and imaginary) values for a total of 256 values. These
are the STFT kernel cooefficients (window function * complex exponential terms).
This prepopulated STFT tensor allows us to not have to recompute the STFT basis
every time we want to process an audio. We can use a convolution using this tensor
to get the frequency spectrogram for the segment and then pass it along to the
encoder blocks.

The second dimension is the number of channels, which for audio is a single
channel assuming mono and not stereo.

The third dimension  256 is the number of frequency output bins that we get from the
STFT. These are the frequencies that are of interest for human speach. For a typical
sampling rate of 16000 Hz, 256 bins gives about 31.25.

Now, for whisper/ggml we need to have the convolution tensor in the format:
```
{256, 1, 258},
```
That is a kernel size of 256, 1 channel, and 258 actual kernels/filters.
```
input [0            639]      // 512 reflection padded with 64 samples

kernel matrix: {256, 1, 258}
0 
  0  [0         255]      [out0_0, out0_1, ...,       out0_255]
1 
  0  [0         255]      [out1_0, out1_1, ...,       out1_255]
2 
  0  [0         255]      [out2_0, out2_1, ...,       out2_255]
  .
  .
  .

257 
  0  [0         255]      [out257_0, out257_1, ..., out255_255]

Output: 
  <----      256 (x-axis)        ---->
  [out0_0, out0_1, ...,       out0_255]   ^
  [out1_0, out1_1, ...,       out1_255]   |
  [out2_0, out2_1, ...,       out2_255]
  .                                       258 (y-axis)
  .                                       
  .                                       |
  [out257_0, out255_1, ..., out255_255]   V

shape: {256, 258, 1}
```
That would be the ouput if we had a stride of 1, but in this case there stride
is 128:
```
input [0..............................................639]
      [0   257] (starts at 0)
               [0   257] (starts at 128)
                        [0   257]  (starts at 256) 
                                 [0   257]  (start at 384)

Output: 
  <----      4 (x-axis)   ---->
  [out0_0, out0_1, out0_2, out0_3]           ^
  [out1_0, out1_1, out1_2, out1_3]           |
  [out2_0, out2_1, out2_3, out2_3]
  .                                          258 (y-axis)
  .                                   
  .                                          |
  [out257_0, out257_1, out257_3, out257_4]   V

shape: {4, 258, 1}
```
Now, the output shape of this operation is:
```console
(gdb) p cur->ne
$13 = {4, 258, 1, 1}
```
So we have a kernel size of 256 and there are 258 actual kernels which contain
the values used in the dot product when we convolve over the input. There are
258 as there are 129 for the cosine/real parts (how much or a specific cosine
frequency exists) and 129 for the sine/imaginary (how much of a specific sine
frequency exists).

So the first layer, `(sftf)` above, will take raw audio samples, 512 samples at
16kHz which is about 32ms.
```
duration = = 1 / 16000 * 512 = 0.032
```

#### Encoder block
The there is an encoder, `(encoder)` above, block which has 4 layers:
```
spectral features → 
Conv1D → ReLU →             Expands to 128 channels
Conv1D → ReLU →             Reduces to 64 channels
Conv1D → ReLU →             Maintains 64 channels
Conv1D → ReLU →             Expands to 128 channels

Kernel size: 3
```
So lets take a look at the first layer:
```
Writing _model.encoder.0.reparam_conv.weight with shape torch.Size([128, 129, 3])

128 output channels
129 input channels
3 kernel size

In ggml this will become a 3D tensor of shape [3, 129, 128]
So this would looks something like this:

0
   0  [0  2]
      ...
 129  [0  2]

...
127
   0  [0  2]
      ...
 129  [0  2]

```

#### Decoder block
Then we have a decoder, `(decoder)` above, block which has 4 layers:
```
encoded features → LSTM Cell → Dropout → ReLU → Conv1D → Sigmoid → speech probability
```
Notice that this is using an LSTM so it is maintaining a hidden state.
The LSTM cell holds state between calls, allowing it to "remember" previous
audio frames. I was a little surprised to see an LSTM here as I read a blog
post prior to looking into Silero-VAD which contained:
```
A few days back we published a new totally reworked Silero VAD. You can try it
on your own voice via interactive demo with a video here or via basic demo here.
We employ a multi-head attention (MHA) based neural network under the hood with
the Short-time Fourier transform as features. This architecture was chosen due
to the fact that MHA-based networks have shown promising results in many
applications ranging from natural language processing to computer vision and
speech processing, but our experiments and recent papers show that you can
achieve good results with any sort of fully feedforward network, you just need
to do enough experiments (i.e. typical choices are MHA-only or transformer
networks, convolutional neural networks or their hybrids) and optimize the
architecture.
```
Perhaps this newer version has not been made available, or have I been looking
at an older version of the model perhaps?  
TODO: Look into this and try to figure out what is going on.

The final sigmoid outputs probability (0-1) of speech presence.

The tensor in the model are the following for the LSTM layer:
```
_model.decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_ih, Shape: torch.Size([512])

_model.decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_hh, Shape: torch.Size([512])
```
The `ih` stands for `input to hidden` and is used to compute the input gate.
Now, notice that the shape is 512, 128 which might seem odd at first but this
actually contains all the vectors for the 4 gates stacked into a matrix.
So we can perform on matrix multiplication to get the input gate.

For the `_model` we have the following parameters:
```
First encoder layer (input 129 frequency bins, output 128 channels, 3 kernel size),
and the bias for that layer:
_model.encoder.0.reparam_conv.weight, Shape: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias, Shape: torch.Size([128])

_model.encoder.1.reparam_conv.weight, Shape: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias, Shape: torch.Size([64])

_model.encoder.2.reparam_conv.weight, Shape: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias, Shape: torch.Size([64])

_model.encoder.3.reparam_conv.weight, Shape: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias, Shape: torch.Size([128])

The decoder LSTM cell has the following parameters:
_model.decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
_model.decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_ih, Shape: torch.Size([512])
_model.decoder.rnn.bias_hh, Shape: torch.Size([512])

Final output layer:
_model.decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias, Shape: torch.Size([1])
```

### Output layer
```
_model.decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias, Shape: torch.Size([1])
```

So, if we start with the raw audio input which consists a samples (floats).
We resample this into either 16kHz or 8kHz, which can be done (at least the 16kHz)
by using `examples/common-whisper.cpp`:
```c++
bool read_audio_data(const std::string & fname,
    std::vector<float>& pcmf32,
    std::vector<std::vector<float>>& pcmf32s,
    bool stereo);
```
One thing to not that this uses `WHISPER_SAMPLE_RATE` which is set to 16000 and
perhaps we should only be focusing on the 16kHz model for now and skip the 8kHz
model?  

So with the output from `read_audio_data` we can then pass this to the VAD.
```c++
std::vector<float> pcmf32;               // mono-channel F32 PCM
std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
```

### Whisper.cpp integration

Branch:  https://github.com/danbev/whisper.cpp/tree/vad

The initial goal is to get the model conversion working and then be able to
load the model and run the graph computation. This currently works and the
test below will run the model and output some results (which don't seem to
be correct).

With this in place I'll start iterating upon this and going through and making
sure that the weights are loaded correctly, and that dimensions for tensors
are correct. Also clean up the code while doing that as I only wanted to get
something working at this stage.

#### Model conversion
To convert silero-vad model first create a virtual environment and install
the version of silero-vad that you want to convert. Then run the conversion:
```console
 $ (venv) pip install silero-vad
 $ (venv) $ python models/convert-silero-vad-to-ggml.py --output models/silero.bin
 Saving GGML Silero-VAD model to models/silero-v5.1.2-ggml.bin

Tensors to be written:
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
_model.encoder.0.reparam_conv.weight: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias: torch.Size([128])
_model.encoder.1.reparam_conv.weight: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias: torch.Size([64])
_model.encoder.2.reparam_conv.weight: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias: torch.Size([64])
_model.encoder.3.reparam_conv.weight: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias: torch.Size([128])
_model.decoder.rnn.weight_ih: torch.Size([512, 128])
_model.decoder.rnn.weight_hh: torch.Size([512, 128])
_model.decoder.rnn.bias_ih: torch.Size([512])
_model.decoder.rnn.bias_hh: torch.Size([512])
_model.decoder.decoder.2.weight: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias: torch.Size([1])

Writing model weights:
Processing variable: _model.encoder.0.reparam_conv.weight with shape: (128, 129, 3)
  Keeping original convolution weight shape: (128, 129, 3)
Processing variable: _model.encoder.0.reparam_conv.bias with shape: (128,)
  Converting to float32
Processing variable: _model.encoder.1.reparam_conv.weight with shape: (64, 128, 3)
  Keeping original convolution weight shape: (64, 128, 3)
Processing variable: _model.encoder.1.reparam_conv.bias with shape: (64,)
  Converting to float32
Processing variable: _model.encoder.2.reparam_conv.weight with shape: (64, 64, 3)
  Keeping original convolution weight shape: (64, 64, 3)
Processing variable: _model.encoder.2.reparam_conv.bias with shape: (64,)
  Converting to float32
Processing variable: _model.encoder.3.reparam_conv.weight with shape: (128, 64, 3)
  Keeping original convolution weight shape: (128, 64, 3)
Processing variable: _model.encoder.3.reparam_conv.bias with shape: (128,)
  Converting to float32
Processing variable: _model.decoder.rnn.weight_ih with shape: (512, 128)
Processing variable: _model.decoder.rnn.weight_hh with shape: (512, 128)
Processing variable: _model.decoder.rnn.bias_ih with shape: (512,)
  Converting to float32
Processing variable: _model.decoder.rnn.bias_hh with shape: (512,)
  Converting to float32
Processing variable: _model.decoder.decoder.2.weight with shape: (128,)
  Converting to float32
Processing variable: _model.decoder.decoder.2.bias with shape: ()
  Converting to float32
Processing variable: _model.stft.forward_basis_buffer with shape: (258, 256)
Done! Model has been converted to GGML format: models/silero-v5.1.2-ggml.bin
```

#### Running Test
Run the test:
```console
$ cmake --build build --target test-vad && \
    ctest -R test-vad --test-dir build --output-on-failure -VV
    ...
10: whisper_vad_init_from_file_with_params_no_state: loading VAD model from '../../models/silero-v5.1.2-ggml.bin'
10: whisper_vad_init_from_file_with_params_no_state: threshold    = 0.500000
10: whisper_vad_init_from_file_with_params_no_state: min_speech_duration_ms = 100
10: whisper_vad_init_from_file_with_params_no_state: min_silence_duration_ms = 100
10: whisper_vad_init_from_file_with_params_no_state: window_size_samples = 512
10: whisper_vad_init_from_file_with_params_no_state: sample_rate = 16000
10: whisper_vad_init_from_file_with_params_no_state: use_f16 = 1
10: whisper_vad_init_from_file_with_params_no_state: n_encoder_layers = 4
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[0] = 129
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[1] = 128
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[2] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[3] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[0] = 128
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[1] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[2] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[3] = 128
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[0] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[1] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[2] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[3] = 3
10: whisper_vad_init_from_file_with_params_no_state: lstm_input_size = 128
10: whisper_vad_init_from_file_with_params_no_state: lstm_hidden_size = 128
10: whisper_vad_init_from_file_with_params_no_state: final_conv_in = 128
10: whisper_vad_init_from_file_with_params_no_state: final_conv_out = 1
10: register_backend: registered backend CPU (1 devices)
10: register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
10: whisper_vad_init_from_file_with_params_no_state:          CPU total size =     0.62 MB
10: whisper_vad_init_from_file_with_params_no_state: model size    =    0.62 MB
10: whisper_backend_init_gpu: no GPU found
10: whisper_vad_build_graph: Building VAD graph
10: whisper_vad_build_encoder_layer: building encoder layer
10: whisper_vad_build_lstm_layer: building LSTM layer
10: whisper_vad_init_state: compute buffer (VAD)   =    1.58 MB
10: whisper_vad_detect_speech: detecting speech in 176000 samples
10: whisper_vad_build_graph: Building VAD graph
10: whisper_vad_build_encoder_layer: building encoder layer
10: whisper_vad_build_lstm_layer: building LSTM layer
10: whisper_vad_detect_speech: window_with_context.size() = 256
10: whisper_vad_detect_speech: window_sample_size: 192
10: whisper_vad_detect_speech: context_sample_size: 64
10: whisper_vad_detect_speech: effective_window_size: 256
10: whisper_vad_detect_speech: frame tensor size: 256
10: whisper_vad_detect_speech: finished processing 176000 samples
10: whisper_vad_detect_speech: prob[0]: 0.030489
10: whisper_vad_detect_speech: prob[1]: 0.020316
10: whisper_vad_detect_speech: prob[2]: 0.016475
10: whisper_vad_detect_speech: prob[3]: 0.011185
10: whisper_vad_detect_speech: prob[4]: 0.010197
10: whisper_vad_detect_speech: prob[5]: 0.007823
10: whisper_vad_detect_speech: prob[6]: 0.008767
10: whisper_vad_detect_speech: prob[7]: 0.006645
10: whisper_vad_detect_speech: prob[8]: 0.005273
10: whisper_vad_detect_speech: prob[9]: 0.010585
10: whisper_vad_detect_speech: prob[10]: 0.007144
10: whisper_vad_detect_speech: prob[11]: 0.003635
10: whisper_vad_detect_speech: prob[12]: 0.004149
10: whisper_vad_detect_speech: prob[13]: 0.005139
10: whisper_vad_detect_speech: prob[14]: 0.003650
10: whisper_vad_detect_speech: prob[15]: 0.007306
10: whisper_vad_detect_speech: prob[16]: 0.004238
10: whisper_vad_detect_speech: prob[17]: 0.004754
10: whisper_vad_detect_speech: prob[18]: 0.003174
10: whisper_vad_detect_speech: prob[19]: 0.001825
10: whisper_vad_detect_speech: prob[20]: 0.005317
10: whisper_vad_detect_speech: prob[21]: 0.004083
10: whisper_vad_detect_speech: prob[22]: 0.002842
10: whisper_vad_detect_speech: prob[23]: 0.004745
```
When I compare this output to the silaro-vad example the values are
very different:
```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
0.9672259092
```
But that was somewhat expected as this was just an attempt to get the model
up and running. Next step will be to go through and figure out where I might
have gotten things wrong.

So lets start by checking that the weights that we are loading are correct.

Lets start with `_model.stft.forward_basis_buffer`:
```console
Original model:
[
0.0,
0.00015059065481182188,
0.0006022718735039234,
0.0013547716662287712,
0.0024076367262750864,
0.003760232590138912,
0.005411745049059391,
0.007361178752034903,
0.009607359766960144,
0.012148935347795486]

GGML model:
```

### Troubleshooting
I started by looking at the tensor `_model.stft.forward_basis_buffer` and printed
out the value from the original model and the whisper.cpp model. The values
from the original model are:
```console
  [0]: 0.0
    [1]: 0.00015059065481182188
    [2]: 0.0006022718735039234
    [3]: 0.0013547716662287712
    [4]: 0.0024076367262750864
    [5]: 0.003760232590138912
    [6]: 0.005411745049059391
    [7]: 0.007361178752034903
    [8]: 0.009607359766960144
    [9]: 0.012148935347795486
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[0]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[1]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[2]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[3]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[4]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[5]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[6]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[7]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[8]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[9]: 0.000000
```
This was because I was not using the correct tensor type. I had make this
configurable to use either `float32` or `float16` but I this will not work with
all operations in GGML. So I've updated the script to for f32 for convolution
operations and after that the values are correct:
```console
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[0]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[1]: 0.000151
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[2]: 0.000602
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[3]: 0.001355
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[4]: 0.002408
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[5]: 0.003760
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[6]: 0.005412
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[7]: 0.007361
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[8]: 0.009607
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[9]: 0.012149
```
But the probabilities are still not the same but I think we can rule out this
tensor (at least how it is read) as the problem here and look at the others.

Now, lets looks at `_model.encoder.0.reparam_conv.weight`
```console
Processing variable: _model.encoder.0.reparam_conv.weight with shape: (128, 129, 3)
  First 10 values for _model.encoder.0.reparam_conv.weight:
    [0]: 0.023059863597154617
    [1]: 0.03755207359790802
    [2]: -0.001536684576421976
    [3]: 0.05659930780529976
    [4]: 0.09177722781896591
    [5]: 0.06459362804889679
    [6]: -0.040349289774894714
    [7]: 0.040909357368946075
    [8]: -0.07200204581022263
    [9]: -0.12808682024478912
  Keeping original convolution weight shape: (128, 129, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
```
And in whisper.cpp:
```console
0: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [0]: 0.023056 (raw: 0x25e7)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [1]: 0.037567 (raw: 0x28cf)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [2]: -0.001536 (raw: 0x964b)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [3]: 0.056610 (raw: 0x2b3f)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [4]: 0.091797 (raw: 0x2de0)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [5]: 0.064575 (raw: 0x2c22)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [6]: -0.040344 (raw: 0xa92a)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [7]: 0.040924 (raw: 0x293d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [8]: -0.072021 (raw: 0xac9c)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [9]: -0.128052 (raw: 0xb019)
```
Lets also check the bias:
```console
Processing variable: _model.encoder.0.reparam_conv.bias with shape: (128,)
  First 10 values for _model.encoder.0.reparam_conv.bias:
    [0]: 0.20333558320999146
    [1]: -0.24448169767856598
    [2]: -2.1663601398468018
    [3]: 0.3871806859970093
    [4]: 0.055092066526412964
    [5]: 0.05976399779319763
    [6]: 0.0019018948078155518
    [7]: 0.8512471914291382
    [8]: -0.11439383029937744
    [9]: -0.0516715943813324
  Original tensor dtype: torch.float32

```
And the bias in whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [0]: 0.203336
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [1]: -0.244482
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [2]: -2.166360
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [3]: 0.387181
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [4]: 0.055092
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [5]: 0.059764
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [6]: 0.001902
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [7]: 0.851247
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [8]: -0.114394
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [9]: -0.051672
```
So these tensors also look correct

Next we have `_model.encoder.1.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.1.reparam_conv.weight with shape: (64, 128, 3)
  First 10 values for _model.encoder.1.reparam_conv.weight:
    [0]: -0.01762554980814457
    [1]: -0.007143480237573385
    [2]: 0.022292815148830414
    [3]: -0.0391620509326458
    [4]: -0.11304397881031036
    [5]: -0.03947301208972931
    [6]: -0.007277275435626507
    [7]: 0.03176437318325043
    [8]: 0.03668201342225075
    [9]: 0.04778497666120529
  Keeping original convolution weight shape: (64, 128, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.1.reparam_conv.bias with shape: (64,)
  First 10 values for _model.encoder.1.reparam_conv.bias:
    [0]: 3.2966432571411133
    [1]: 1.6271023750305176
    [2]: -7.954858779907227
    [3]: 2.7928881645202637
    [4]: 0.10639765858650208
    [5]: 1.5769203901290894
    [6]: 1.2196542024612427
    [7]: 1.5114142894744873
    [8]: 0.9804346561431885
    [9]: -7.94569206237793
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [0]: -0.017624 (raw: 0xa483)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [1]: -0.007145 (raw: 0x9f51)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [2]: 0.022293 (raw: 0x25b5)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [3]: -0.039154 (raw: 0xa903)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [4]: -0.113037 (raw: 0xaf3c)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [5]: -0.039459 (raw: 0xa90d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [6]: -0.007278 (raw: 0x9f74)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [7]: 0.031769 (raw: 0x2811)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [8]: 0.036682 (raw: 0x28b2)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [9]: 0.047791 (raw: 0x2a1e)
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [0]: 3.296643
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [1]: 1.627102
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [2]: -7.954859
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [3]: 2.792888
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [4]: 0.106398
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [5]: 1.576920
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [6]: 1.219654
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [7]: 1.511414
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [8]: 0.980435
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [9]: -7.945692
```
The look correct as well.

Then we have `_model.encoder.2.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.2.reparam_conv.weight with shape: (64, 64, 3)
  First 10 values for _model.encoder.2.reparam_conv.weight:
    [0]: -0.0072915456257760525
    [1]: -0.10136377811431885
    [2]: -0.19760535657405853
    [3]: -0.0005110583733767271
    [4]: -0.01200706698000431
    [5]: -0.0048386408016085625
    [6]: -0.006183745805174112
    [7]: 0.07137007266283035
    [8]: 0.05046859756112099
    [9]: -0.003160792402923107
  Keeping original convolution weight shape: (64, 64, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.2.reparam_conv.bias with shape: (64,)
  First 10 values for _model.encoder.2.reparam_conv.bias:
    [0]: 4.060866832733154
    [1]: 3.816256523132324
    [2]: 0.053663045167922974
    [3]: 0.9439471960067749
    [4]: 2.875575065612793
    [5]: 0.27411338686943054
    [6]: 0.8237091302871704
    [7]: -1.587329626083374
    [8]: -0.9315840005874634
    [9]: 1.7247822284698486
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [0]: -0.007290 (raw: 0x9f77)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [1]: -0.101379 (raw: 0xae7d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [2]: -0.197632 (raw: 0xb253)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [3]: -0.000511 (raw: 0x9030)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [4]: -0.012009 (raw: 0xa226)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [5]: -0.004837 (raw: 0x9cf4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [6]: -0.006184 (raw: 0x9e55)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [7]: 0.071350 (raw: 0x2c91)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [8]: 0.050476 (raw: 0x2a76)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [9]: -0.003160 (raw: 0x9a79)
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [0]: 4.060867
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [1]: 3.816257
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [2]: 0.053663
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [3]: 0.943947
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [4]: 2.875575
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [5]: 0.274113
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [6]: 0.823709
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [7]: -1.587330
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [8]: -0.931584
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [9]: 1.724782
```
And these look correct as well.

The we have `_model.encoder.3.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.3.reparam_conv.weight with shape: (128, 64, 3)
  First 10 values for _model.encoder.3.reparam_conv.weight:
    [0]: 0.00868716835975647
    [1]: -0.08090031892061234
    [2]: 0.01122092455625534
    [3]: 0.0034291022457182407
    [4]: 0.023257968947291374
    [5]: 0.008206821046769619
    [6]: 0.006397297605872154
    [7]: 0.18601815402507782
    [8]: 0.007254657801240683
    [9]: -0.0012539586750790477
  Keeping original convolution weight shape: (128, 64, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.3.reparam_conv.bias with shape: (128,)
  First 10 values for _model.encoder.3.reparam_conv.bias:
    [0]: 0.9335513114929199
    [1]: 0.11157345771789551
    [2]: 0.09006297588348389
    [3]: 0.6109893918037415
    [4]: -0.6373689770698547
    [5]: 0.00609125941991806
    [6]: 1.0473954677581787
    [7]: -0.6057872176170349
    [8]: 1.885377049446106
    [9]: -3.769871711730957
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [0]: 0.008690 (raw: 0x2073)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [1]: -0.080872 (raw: 0xad2d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [2]: 0.011223 (raw: 0x21bf)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [3]: 0.003429 (raw: 0x1b06)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [4]: 0.023254 (raw: 0x25f4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [5]: 0.008209 (raw: 0x2034)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [6]: 0.006397 (raw: 0x1e8d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [7]: 0.186035 (raw: 0x31f4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [8]: 0.007256 (raw: 0x1f6e)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [9]: -0.001254 (raw: 0x9523)
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [0]: 0.933551
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [1]: 0.111573
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [2]: 0.090063
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [3]: 0.610989
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [4]: -0.637369
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [5]: 0.006091
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [6]: 1.047395
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [7]: -0.605787
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [8]: 1.885377
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [9]: -3.769872
```
And these also look correct. 

Next lets check the LSTM/RNN tensors:
```console
Processing variable: _model.decoder.rnn.weight_ih with shape: (512, 128)
  First 10 values for _model.decoder.rnn.weight_ih:
    [0]: -0.1975371241569519
    [1]: -0.13793830573558807
    [2]: 0.16510847210884094
    [3]: 0.007955566048622131
    [4]: 0.029819002375006676
    [5]: -0.3347293436527252
    [6]: 0.019417593255639076
    [7]: 0.00517271226271987
    [8]: -0.08036171644926071
    [9]: 0.14333027601242065
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.rnn.bias_ih with shape: (512,)
  First 10 values for _model.decoder.rnn.bias_ih:
    [0]: -0.1524425894021988
    [1]: -0.12193526327610016
    [2]: -0.08168794959783554
    [3]: -0.29849109053611755
    [4]: -0.2474878579378128
    [5]: 0.03450224548578262
    [6]: -0.08904067426919937
    [7]: -0.06718937307596207
    [8]: -0.12373599410057068
    [9]: -0.392291396856308
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [0]: -0.197537
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [1]: -0.137938
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [2]: 0.165108
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [3]: 0.007956
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [4]: 0.029819
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [5]: -0.334729
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [6]: 0.019418
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [7]: 0.005173
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [8]: -0.080362
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [9]: 0.143330

10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [0]: -0.152443
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [1]: -0.121935
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [2]: -0.081688
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [3]: -0.298491
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [4]: -0.247488
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [5]: 0.034502
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [6]: -0.089041
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [7]: -0.067189
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [8]: -0.123736
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [9]: -0.392291
```
These look correct as well (apart for an inconsistency in the nameing of the
tensor in whisper.cpp, I'll fix that).

Next we have `_model.decoder.rnn.weight_hh`:
```console
Processing variable: _model.decoder.rnn.weight_hh with shape: (512, 128)
  First 10 values for _model.decoder.rnn.weight_hh:
    [0]: -0.3621460497379303
    [1]: 0.14502376317977905
    [2]: -0.29783394932746887
    [3]: 0.034422460943460464
    [4]: 0.17480415105819702
    [5]: -0.1250990778207779
    [6]: -0.24738839268684387
    [7]: -0.06837962567806244
    [8]: 0.32639244198799133
    [9]: -0.18058985471725464
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.rnn.bias_hh with shape: (512,)
  First 10 values for _model.decoder.rnn.bias_hh:
    [0]: -0.023373831063508987
    [1]: -0.13415886461734772
    [2]: -0.04436622932553291
    [3]: -0.4029233157634735
    [4]: -0.23194685578346252
    [5]: -0.01958276331424713
    [6]: -0.03060426004230976
    [7]: -0.03582705929875374
    [8]: -0.17606812715530396
    [9]: -0.2881392538547516
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [0]: -0.362146
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [1]: 0.145024
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [2]: -0.297834
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [3]: 0.034422
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [4]: 0.174804
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [5]: -0.125099
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [6]: -0.247388
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [7]: -0.068380
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [8]: 0.326392
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [9]: -0.180590

10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [0]: -0.023374
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [1]: -0.134159
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [2]: -0.044366
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [3]: -0.402923
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [4]: -0.231947
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [5]: -0.019583
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [6]: -0.030604
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [7]: -0.035827
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [8]: -0.176068
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [9]: -0.288139
````
And these look correct as well.

And finally we have `_model.decoder.decoder.2.weight` and bias:
```console
Processing variable: _model.decoder.decoder.2.weight with shape: (128,)
  First 10 values for _model.decoder.decoder.2.weight:
    [0]: 0.10062672197818756
    [1]: 0.17330233752727509
    [2]: -0.251087486743927
    [3]: -1.1117055416107178
    [4]: 0.30843374133110046
    [5]: -0.44464311003685
    [6]: -0.45811617374420166
    [7]: -0.027409639209508896
    [8]: 0.3915608525276184
    [9]: 1.2692075967788696
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.decoder.2.bias with shape: ()
  First 10 values for _model.decoder.decoder.2.bias:
    [0]: -0.19063705205917358
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [0]: 0.100627
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [1]: 0.173302
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [2]: -0.251087
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [3]: -1.111706
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [4]: 0.308434
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [5]: -0.444643
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [6]: -0.458116
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [7]: -0.027410
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [8]: 0.391561
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [9]: 1.269208

10: whisper_vad_init_from_file_with_params_no_state: final_conv_bias: [0]: -0.190637
```
So the weight seem to be correct for this as well.

Looking little more carefully at the origal model above I noticed a mistake
I've made.
```
    (decoder): RecursiveScriptModule(
      original_name=VADDecoderRNNJIT
      (rnn): RecursiveScriptModule(original_name=LSTMCell)
      (decoder): RecursiveScriptModule(
        original_name=Sequential
        (0): RecursiveScriptModule(original_name=Dropout)
        (1): RecursiveScriptModule(original_name=ReLU)
        (2): RecursiveScriptModule(original_name=Conv1d)
        (3): RecursiveScriptModule(original_name=Sigmoid)
      )
    )
```
This shows a the LSTM layer, followed by a dropout, a relu, and conv1d and then
a sigmoid. I did not include a drop out or a ReLU, and also instead of a conv1d I
has a simple linear layer (mul_mat plus add bias). This is a mistake, and I've
now updated with these layers.

With those changes in place I get the following probabilities:
```console
10: whisper_vad_detect_speech: prob[0]: 0.433458
10: whisper_vad_detect_speech: prob[1]: 0.231823
10: whisper_vad_detect_speech: prob[2]: 0.180998
10: whisper_vad_detect_speech: prob[3]: 0.151418
10: whisper_vad_detect_speech: prob[4]: 0.129234
10: whisper_vad_detect_speech: prob[5]: 0.118154
10: whisper_vad_detect_speech: prob[6]: 0.117612
10: whisper_vad_detect_speech: prob[7]: 0.103644
10: whisper_vad_detect_speech: prob[8]: 0.100053
10: whisper_vad_detect_speech: prob[9]: 0.095149
10: whisper_vad_detect_speech: prob[10]: 0.081977
10: whisper_vad_detect_speech: prob[11]: 0.080713
10: whisper_vad_detect_speech: prob[12]: 0.083069
10: whisper_vad_detect_speech: prob[13]: 0.081101
10: whisper_vad_detect_speech: prob[14]: 0.081165
10: whisper_vad_detect_speech: prob[15]: 0.069563
10: whisper_vad_detect_speech: prob[16]: 0.059652
10: whisper_vad_detect_speech: prob[17]: 0.060627
10: whisper_vad_detect_speech: prob[18]: 0.060169
```
Which are better but still not the same as the original model.
```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
```

Now, I noticed another thing that in the original model and the python example
and also the C++ example they use a window size of 512 where I've been using
192, which was basically just to allow the matrix multiplication to work with
the stft matrix. Stepping back a little this is how I understand the process
up until this point:
We have samples, in this specific case around 17000 samples. These are split
into chunks for 512 (which if I recall correctly is about 32ms and close to the
threshold of human perception of sound). We add an extra 64 samples which we
take from the previous window to avoid spectral leakage. Now we want to multiple
these sample by the matrix provided by the silero-vad model which contains
precomputed sine and cosine values. But doing this we get the values in the
frequency domain (as complex numbers but we only care about the real part here).
Now, this matrix has the shape {256, 258}, and the frame tensor is {576} and
these cannot be multiplied with each other.

I took a look at the ONNX runtime and it seems to be implementing this as a
convolution operation and not a matrix multiplication which was what I did.

Here is an excert from the onnxruntime log:
```console
   20 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_before","args" : {"op_name" : "Pad"}},
   21 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "32","activation_size" : "2304","output_t      ype_shape" : [{"float":[1,640]}],"exec_plan_index" : "7","graph_index" : "7","input_type_shape" : [{"float":[1,576]},{"int64":[4]}],"provider" : "CPUExecutionProvider","op_name" : "Pad"}},
   22 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_after","args" : {"op_name" : "Pad"}},
   23 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_before","args" : {"op_name" : "Unsqueeze"}},
   24 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59313,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "8","activation_size" : "2560","output_type      _shape" : [{"float":[1,1,640]}],"exec_plan_index" : "8","graph_index" : "8","input_type_shape" : [{"float":[1,640]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Unsqueeze"}},
   25 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59317,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_after","args" : {"op_name" : "Unsqueeze"}},
   26 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_before","args" : {"op_name" : "Conv"}},
   27 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :54,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "4128","parameter_size" : "264192","activation_size" : "2560","output_typ      e_shape" : [{"float":[1,258,4]}],"exec_plan_index" : "9","graph_index" : "9","input_type_shape" : [{"float":[1,1,640]},{"float":[258,1,256]}],"provider" : "CPUExecutionProvider","op_name" : "Conv"}},
   28 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59375,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_after","args" : {"op_name" : "Conv"}},
   29 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59376,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_before","args" : {"op_name" : "Slice"}},
   30 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59377,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "14","graph_index" : "14","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   31 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59385,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_after","args" : {"op_name" : "Slice"}},
   32 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_before","args" : {"op_name" : "Pow"}},
   33 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_sha      pe" : [{"float":[1,129,4]}],"exec_plan_index" : "18","graph_index" : "18","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   34 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59391,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_after","args" : {"op_name" : "Pow"}},
   35 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_before","args" : {"op_name" : "Slice"}},
   36 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "11","graph_index" : "11","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   37 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59398,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_after","args" : {"op_name" : "Slice"}},
   38 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_before","args" : {"op_name" : "Pow"}},
   39 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "17","graph_index" : "17","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   40 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59403,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_after","args" : {"op_name" : "Pow"}},
   41 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_before","args" : {"op_name" : "Add"}},
   42 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "4128","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "19","graph_index" : "19","input_type_shape" : [{"float":[1,129,4]},{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Add"}},
   43 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59409,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_after","args" : {"op_name" : "Add"}},
   44 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59410,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_fence_before","args" : {"op_name" : "Sqrt"}},
   45 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59411,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "2064","output_type_shap      e" : [{"float":[1,129,4]}],"exec_plan_index" : "20","graph_index" : "20","input_type_shape" : [{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Sqrt"}},
   46 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59418,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_fence_after","args" : {"op_name" : "Sqrt"}},
```
So after consulting Claude.ai about this is looks like ONNX is performing a
convolution operation. So I tried this and the results were the exact same I
think:
```console
10: whisper_vad_detect_speech: prob[0]: 0.433458
10: whisper_vad_detect_speech: prob[1]: 0.231823
10: whisper_vad_detect_speech: prob[2]: 0.180998
10: whisper_vad_detect_speech: prob[3]: 0.151459
10: whisper_vad_detect_speech: prob[4]: 0.129263
10: whisper_vad_detect_speech: prob[5]: 0.115693
10: whisper_vad_detect_speech: prob[6]: 0.106706
10: whisper_vad_detect_speech: prob[7]: 0.101227
10: whisper_vad_detect_speech: prob[8]: 0.097039
10: whisper_vad_detect_speech: prob[9]: 0.093106
10: whisper_vad_detect_speech: prob[10]: 0.088892
10: whisper_vad_detect_speech: prob[11]: 0.084535
10: whisper_vad_detect_speech: prob[12]: 0.079914
10: whisper_vad_detect_speech: prob[13]: 0.075425
```

So I'm getting the following output at the moment:
```console
10: whisper_vad_detect_speech: prob[0]: 0.017022
10: whisper_vad_detect_speech: prob[1]: 0.018527
10: whisper_vad_detect_speech: prob[2]: 0.011746
10: whisper_vad_detect_speech: prob[3]: 0.008625
10: whisper_vad_detect_speech: prob[4]: 0.004357
10: whisper_vad_detect_speech: prob[5]: 0.003329
10: whisper_vad_detect_speech: prob[6]: 0.002859
10: whisper_vad_detect_speech: prob[7]: 0.005444
10: whisper_vad_detect_speech: prob[8]: 0.007293
10: whisper_vad_detect_speech: prob[9]: 0.004256
```
Compared to the output from the silero-vad pythons example:
```console
0.012012
0.010678
0.132181
0.065489
0.044598
0.022335
0.026070
0.011671
0.008116
0.006716
```
One thing I noticed is that if I comment out the actual setting of the pcm32
data, I get the same result.
```c++
        // Copy the current samples from pcmf32 into the window_with_context,
        // starting after the context buffer copied above.
        //std::copy(&pcmf32[i], &pcmf32[i + vctx->window_size_samples], window_with_context.begin() + vctx->context_samples);
```

So it is like the probabilities I'm seeing are
just noise.

```console
10: whisper_vad_detect_speech: prob[0]: 0.017022
10: whisper_vad_detect_speech: prob[1]: 0.018406
10: whisper_vad_detect_speech: prob[2]: 0.051465
10: whisper_vad_detect_speech: prob[3]: 0.028821
10: whisper_vad_detect_speech: prob[4]: 0.012965
10: whisper_vad_detect_speech: prob[5]: 0.022604
10: whisper_vad_detect_speech: prob[6]: 0.056287
10: whisper_vad_detect_speech: prob[7]: 0.020504
10: whisper_vad_detect_speech: prob[8]: 0.015851
10: whisper_vad_detect_speech: prob[9]: 0.018630
10: whisper_vad_detect_speech: prob[10]: 0.037504
10: whisper_vad_detect_speech: prob[11]: 0.141271
10: whisper_vad_detect_speech: prob[12]: 0.051602
10: whisper_vad_detect_speech: prob[13]: 0.029109
10: whisper_vad_detect_speech: prob[14]: 0.110565
10: whisper_vad_detect_speech: prob[15]: 0.047791
10: whisper_vad_detect_speech: prob[16]: 0.031876
10: whisper_vad_detect_speech: prob[17]: 0.086297
10: whisper_vad_detect_speech: prob[18]: 0.041629
10: whisper_vad_detect_speech: prob[19]: 0.097479
10: whisper_vad_detect_speech: prob[20]: 0.073999
10: whisper_vad_detect_speech: prob[21]: 0.063608
10: whisper_vad_detect_speech: prob[22]: 0.078973
10: whisper_vad_detect_speech: prob[23]: 0.486158
10: whisper_vad_detect_speech: prob[24]: 0.609635
10: whisper_vad_detect_speech: prob[25]: 0.028430
```

Inpsecting the silero-vad model a little closer I found:
```console
5030         node {
5031           input: "If_0_then_branch__Inline_0__/stft/Unsqueeze_output_0"
5032           input: "If_0_then_branch__Inline_0__stft.forward_basis_buffer"
5033           output: "If_0_then_branch__Inline_0__/stft/Conv_output_0"
5034           name: "If_0_then_branch__Inline_0__/stft/Conv"
5035           op_type: "Conv"
5036           attribute {
5037             name: "dilations"
5038             ints: 1
5039             type: INTS
5040           }
5041           attribute {
5042             name: "group"
5043             i: 1
5044             type: INT
5045           }
5046           attribute {
5047             name: "kernel_shape"
5048             ints: 256
5049             type: INTS
5050           }
5051           attribute {
5052             name: "pads"
5053             ints: 0
5054             ints: 0
5055             type: INTS
5056           }
5057           attribute {
5058             name: "strides"
5059             ints: 128
5060             type: INTS
5061           }
5062         }
```
So it looks like I was using incorrect padding and stride values. I've updated
this now and the results are much the same. But I'm still only able to inspect
the final probabilities and I don't know for sure if the sftf values are correct
or not. I need to be able to print the intermediate values from the onnx runtime
and compare them. 

After doing throught model once more and looking that the shapes for the
dimensions and stride for some of the convolution operations that I was missing
the probabilities are now:
```console
10: whisper_vad_detect_speech: prob[0]: 0.001881
10: whisper_vad_detect_speech: prob[1]: 0.001317
10: whisper_vad_detect_speech: prob[2]: 0.008114
10: whisper_vad_detect_speech: prob[3]: 0.002658
10: whisper_vad_detect_speech: prob[4]: 0.000671
10: whisper_vad_detect_speech: prob[5]: 0.000244
10: whisper_vad_detect_speech: prob[6]: 0.000759
10: whisper_vad_detect_speech: prob[7]: 0.005907
10: whisper_vad_detect_speech: prob[8]: 0.005715
10: whisper_vad_detect_speech: prob[9]: 0.005150
```

```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
0.9672259092
```

### Mimic the silero-vad model

I've not been able find an good way of being able to print intermediate values
in the silero-vad model so I resorted creating a PyTorch implementation, and
then converting the model, similar to how we convert the model to ggml format.
```console
$ cd audio/silero-vad
(venv) $ python src/reverse-eng/extract_conv_weights.py
Loading JIT model from /home/danbev/work/ai/audio/silero-vad/src/silero_vad/data/silero_vad.jit
Found 30 parameters in JIT model
Transferred: _model.stft.forward_basis_buffer -> stft.forward_basis_buffer, Shape: torch.Size([258, 1, 256])
Transferred: _model.encoder.0.reparam_conv.weight -> encoder.0.reparam_conv.weight, Shape: torch.Size([128, 129, 3])
Transferred: _model.encoder.0.reparam_conv.bias -> encoder.0.reparam_conv.bias, Shape: torch.Size([128])
Transferred: _model.encoder.1.reparam_conv.weight -> encoder.1.reparam_conv.weight, Shape: torch.Size([64, 128, 3])
Transferred: _model.encoder.1.reparam_conv.bias -> encoder.1.reparam_conv.bias, Shape: torch.Size([64])
Transferred: _model.encoder.2.reparam_conv.weight -> encoder.2.reparam_conv.weight, Shape: torch.Size([64, 64, 3])
Transferred: _model.encoder.2.reparam_conv.bias -> encoder.2.reparam_conv.bias, Shape: torch.Size([64])
Transferred: _model.encoder.3.reparam_conv.weight -> encoder.3.reparam_conv.weight, Shape: torch.Size([128, 64, 3])
Transferred: _model.encoder.3.reparam_conv.bias -> encoder.3.reparam_conv.bias, Shape: torch.Size([128])
Transferred: _model.decoder.rnn.weight_ih -> decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
Transferred: _model.decoder.rnn.weight_hh -> decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
Transferred: _model.decoder.rnn.bias_ih -> decoder.rnn.bias_ih, Shape: torch.Size([512])
Transferred: _model.decoder.rnn.bias_hh -> decoder.rnn.bias_hh, Shape: torch.Size([512])
Transferred: _model.decoder.decoder.2.weight -> decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
Transferred: _model.decoder.decoder.2.bias -> decoder.decoder.2.bias, Shape: torch.Size([1])
Saved PyTorch model to silero_vad_conv_pytorch.pth
Weight extraction complete!
```
With the model extracted we can now run an single inference and compare it to
values that we produce and to what the original model produces:
```console
(venv) $ python src/reverse-eng/test_conv_model_with_weights.py
Loading PyTorch model from silero_vad_conv_pytorch.pth
STFT basis buffer samples: tensor([[0.0000, 0.0002, 0.0006, 0.0014, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0014, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0013, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0013, 0.0023],
        [0.0000, 0.0001, 0.0006, 0.0013, 0.0022]])
Input tensor shape: torch.Size([1, 512])
STFT basis buffer samples: tensor([[0.0000, 0.0002, 0.0006, 0.0014, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0014, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0013, 0.0024],
        [0.0000, 0.0002, 0.0006, 0.0013, 0.0023],
        [0.0000, 0.0001, 0.0006, 0.0013, 0.0022]])
shape of STFT output: torch.Size([1, 258, 4])
Full STFT (before slicing): tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

Intermediate shapes from PyTorch model:
  input: torch.Size([1, 512])
    Sample values: tensor([0., 0., 0., 0., 0.])
  stft_out: torch.Size([1, 129, 4])
    Sample values: tensor([0., 0., 0., 0., 0.])
  encoder_out: torch.Size([1, 128, 1])
    Sample values: tensor([0.0000, 0.2486, 1.7993, 0.0000, 0.3163])
  features: torch.Size([1, 128])
    Sample values: tensor([0.0000, 0.2486, 1.7993, 0.0000, 0.3163])
  lstm_h: torch.Size([1, 128])
    Sample values: tensor([0.0037, 0.2473, 0.0583, 0.2980, 0.0026])
  lstm_c: torch.Size([1, 128])
    Sample values: tensor([0.0041, 0.2626, 0.0706, 0.3206, 0.0569])

PyTorch output: 0.001875
Saved activation visualization to conv_model_activations.png
```
And then compare this to the values that whisper.cpp produces (also just one
sample):
```console
$ ./test-vad.sh
10: whisper_vad_detect_speech: sftf shape: {4, 258, 1}
10: whisper_vad_detect_speech: sftf: [0]: -2.878450
10: whisper_vad_detect_speech: sftf: [1]: 0.008860
10: whisper_vad_detect_speech: sftf: [2]: -2.506309
10: whisper_vad_detect_speech: sftf: [3]: -0.403049
10: whisper_vad_detect_speech: sftf: [4]: -2.481766
10: whisper_vad_detect_speech: sftf: [5]: -2.231346
10: whisper_vad_detect_speech: sftf: [6]: 1.069640
10: whisper_vad_detect_speech: sftf: [7]: -0.406165
10: whisper_vad_detect_speech: sftf: [8]: -1.768491
10: whisper_vad_detect_speech: sftf: [9]: -5.376316
10: whisper_vad_detect_speech: final_conv: [0]: -6.274172
10: whisper_vad_detect_speech: h_state first 3 values: 0.003820, 0.247065, 0.058351
10: whisper_vad_detect_speech: c_state first 3 values: 0.004212, 0.262400, 0.070621
10: whisper_vad_detect_speech: finished processing 176000 samples
10: whisper_vad_detect_speech: prob[0]: 0.001881
```
Hopefully by doing this and comparing step by step I can try to figure out where
I'm going wrong. So the idea is to get the these two implementation to produce
the same output as the original model.
Notice the difference in the output here for the intermediate values of the
stft operation. In the python version it is all zeros while in whisper.cpp the
have very different values. This is becuase whisper.cpp is using a schduler and
this means that tensors can be reused for different operations as an
optimization. If we instead create a copy of the tensor:
```c++
    struct ggml_tensor * stft_copy = ggml_conv_1d(ctx0, stft_reshaped, padded, 128, 0, 1);
    ggml_set_name(stft_copy, "stft_copy");
    ggml_set_output(stft_copy);
    ggml_build_forward_expand(gf, stft_copy);
```
And print out that tensor instead we get:
```console
10: whisper_vad_detect_speech: sftf: [0]: 0.000000
10: whisper_vad_detect_speech: sftf: [1]: 0.000000
10: whisper_vad_detect_speech: sftf: [2]: 0.000000
10: whisper_vad_detect_speech: sftf: [3]: 0.000000
10: whisper_vad_detect_speech: sftf: [4]: 0.000000
10: whisper_vad_detect_speech: sftf: [5]: 0.000000
10: whisper_vad_detect_speech: sftf: [6]: 0.000000
10: whisper_vad_detect_speech: sftf: [7]: 0.000000
10: whisper_vad_detect_speech: sftf: [8]: 0.000000
10: whisper_vad_detect_speech: sftf: [9]: 0.000000
```
I need to keep this in mind when debugging!

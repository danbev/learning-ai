## Core ML
Is an Apple framework that allows developers to integrate machine learning models into their apps.

## Core ML Model
Is like a bundle, so it is a directory that can store a model in separate files and
also includes metadata about the model. The model is stored in a file with the extension
`.mlmodel` and the metadata is stored in a file with the extension `.mlmodeldescription`.

## Whisper.cpp and Core ML
Whisper can use Core ML...
```c++
struct whisper_context * whisper_init_from_file_with_params(const char * path_model, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_state * whisper_init_state(whisper_context * ctx) {
    whisper_state * state = new whisper_state;
    ...

#ifdef WHISPER_USE_COREML
    const auto path_coreml = whisper_get_coreml_path_encoder(ctx->path_model);

    WHISPER_LOG_INFO("%s: loading Core ML model from '%s'\n", __func__, path_coreml.c_str());
    WHISPER_LOG_INFO("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_coreml = whisper_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        WHISPER_LOG_ERROR("%s: failed to load Core ML model from '%s'\n", __func__, path_coreml.c_str());
#ifndef WHISPER_COREML_ALLOW_FALLBACK
        whisper_free_state(state);
        return nullptr;
#endif
    } else {
        WHISPER_LOG_INFO("%s: Core ML model loaded\n", __func__);
    }
#endif
}
```
```console
(lldb) p ctx->path_model
(std::string) "models/ggml-base.en.bin"
```
```c++
#ifdef WHISPER_USE_COREML
// replace .bin with -encoder.mlmodelc
static std::string whisper_get_coreml_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    // match "-qx_x"
    pos = path_bin.rfind('-');
    if (pos != std::string::npos) {
        auto sub = path_bin.substr(pos);
        if (sub.size() == 5 && sub[1] == 'q' && sub[3] == '_') {
            path_bin = path_bin.substr(0, pos);
        }
    }

    path_bin += "-encoder.mlmodelc";

    return path_bin;
}
#endif
```
```console
(lldb) p path_bin
(std::string) "models/ggml-base.en-encoder.mlmodelc"
```
Now, lets take a closer look at `whisper_coreml_init` which can be found in `src/coreml/whisper-encoder.mm`:
```c++
struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    //config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.computeUnits = MLComputeUnitsAll;

    const void * data = CFBridgingRetain([[whisper_encoder_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    whisper_coreml_context * ctx = new whisper_coreml_context;

    ctx->data = data;

    return ctx;
}
```
First a configuration object is created:
```console
(lldb) p *config
(MLModelConfiguration) {
  NSObject = {
    isa = MLModelConfiguration
  }
  _experimentalMLE5EngineUsage = 0
  _usePrecompiledE5Bundle = false
  _experimentalMLE5BNNSGraphBackendUsage = 0
  _experimentalMLE5BNNSGraphBackendUsageMultiSegment = 0
  _e5rtDynamicCallableFunctions = 0x00000001f882a230
  _e5rtMutableMILWeightURLs = 0x00000001f882a230
  _e5rtComputeDeviceTypeMask = 18446744073709551615
  _e5rtCustomANECompilerOptions = nil
  _serializesMILTextForDebugging = false
  _specializationUsesMPSGraphExecutable = true
  _allowBackgroundGPUComputeSetting = false
  _trainWithMLCompute = false
  _useWatchSPIForScribble = false
  _allowLowPrecisionAccumulationOnGPU = false
  _enableTestVectorMode = false
  _usePreloadedKey = false
  _allowsInstrumentation = true
  _preparesLazily = false
  _modelDisplayName = nil
  _computeUnits = 2
  _optimizationHints = 0x0000600001138360
  _functionName = nil
  _predictionConcurrencyHint = 0
  _preferredMetalDevice = nil
  _parameters = nil
  _rootModelURL = nil
  _profilingOptions = 0
  _parentModelName = 0x00000001f921a1f8 ""
}
````
The code is then setting the `computeUnits` property to `MLComputeUnitsAll` which allows the
OS to select the best device to run the model on:
```c++
    config.computeUnits = MLComputeUnitsAll;
```
Then we have;
```c++
    const void * data = CFBridgingRetain([[whisper_encoder_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);
```
This is converting the `whisper_encoder_impl` object to a `void` pointer and then retaining it. This allows
the object to be safely stored and used from c++ code.
We can see the definition of `whisper_encoder_impl` in `src/coreml/whisper-encoder-impl.h`:
```c++
/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface whisper_encoder_impl : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;
```
So this only has a single property/member which is a `MLModel` object.
We can inspect the  MLModel object in the debugger:
```console
(lldb) expression -l objc -O -- [ctx->data model]

modelDescription:
functionName: (null)
inputs: (
    "logmel_data : MultiArray (Float32, 1 \U00d7 80 \U00d7 3000)"
)
outputs: (
    "output : MultiArray (Float32, )"
)
states: (
)
predictedFeatureName: (null)
predictedProbabilitiesName: (null)
classLabels: (null)
updatable: NO
trainingInputs: (
)
parameters: (
)
metadata: {
    MLModelAuthorKey = "";
    MLModelCreatorDefinedKey =     {
        "com.github.apple.coremltools.source" = "torch==2.6.0";
        "com.github.apple.coremltools.source_dialect" = TorchScript;
        "com.github.apple.coremltools.version" = "8.2";
    };
    MLModelDescriptionKey = "";
    MLModelLicenseKey = "";
    MLModelVersionStringKey = "";
},
configuration:
 computeUnits: All,
 useWatchSPIForScribble: NO,
 allowLowPrecisionAccumulationOnGPU: NO,
 allowBackgroundGPUComputeSetting: NO,
 preferredMetalDevice: (null),
 enableTestVectorMode: NO,
 parameters: (null),
 rootModelURL: file:///Users/danbev/work/ai/whisper.cpp/models/ggml-base.en-encoder.mlmodelc/,
 profilingOptions: 0,
 usePreloadedKey: NO,
 trainWithMLCompute: NO,
 parentModelName: ,
 modelName: ggml-base.en-encoder,
 experimentalMLE5EngineUsage: Enable,
 preparesLazily: NO,
 predictionConcurrencyHint: 0,
 usePrecompiledE5Bundle: NO,
 experimentalMLE5BNNSGraphBackendUsage: Enable,
 e5rtMutableMILWeightURLs: {
}
 e5rtDynamicCallableFunctions: {
}
 optimizationHints: MLOptimizationHints:
 reshapeFrequency Frequent
 hotHandDuration 1.00 seconds
 specializationStrategy Default
,
 functionName: (null),
 experimentalMLE5BNNSGraphBackendUsageMultiSegment: Enable,
 e5rtComputeDeviceTypeMask: 0xffffffffffffffff,
 e5rtCustomANECompilerOptions: (null),
 serializesMILTextForDebugging: NO,
 specializationUsesMPSGraphExecutable: YES
 ```
Notice that the input to this model is the log mel spectrogram with dimensions 1x80x3000:
```console
inputs: (
    "logmel_data : MultiArray (Float32, 1 \U00d7 80 \U00d7 3000)"
)
```
So whisper.cpp will take care of this "pre-processing" and then pass that data to coreml
that will perform the encoding part of the inference process. The decoding is still done
by whipser.cpp if I've understood things correctly.

Later in `whisper_encode_internal` we have:
```c++
static bool whisper_encode_internal(
        whisper_context & wctx,
          whisper_state & wstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data) {
    ...
        if (!whisper_encode_external(wstate)) {
            if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
                return false;
            }
        } else {
#if defined(WHISPER_USE_COREML)
            whisper_coreml_encode(wstate.ctx_coreml, mel->ne[0], mel->ne[1], (float *) mel->data, (float *) wstate.embd_enc->data);
#elif defined(WHISPER_USE_OPENVINO)
            whisper_openvino_encode(wstate.ctx_openvino, mel, wstate.embd_enc);
#endif
        }
```
So this is checking if encode is external, as in coreml or openvino. In our case this will
call `whisper_coreml_encode`:
```c++
void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                               float * mel,
                               float * out) {
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: mel
                                           shape: @[@1, @(n_mel), @(n_ctx)]
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx*n_mel), @(n_ctx), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromLogmel_data:inMultiArray error:nil];

        memcpy(out, outCoreML.output.dataPointer, outCoreML.output.count * sizeof(float));
    }
}
```
`MLMultiArray` is a class that represents Core ML's tensor data. It will use the pointer to the mel 
data.
```console
(lldb) expression -l objc -O -- [ctx->data model]
```

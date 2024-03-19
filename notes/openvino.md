## Open Visual Inference & Neural Network Optimization (OpenVINO)
Is a toolkit developed by Intel and it for developing and acceleration of
applications that require computer vision and deep learning inference.

So this is a framework that works accross Intel hardware (CPU, GPU, FPGA ect).

### Installing OpenVINO
Follow the [instructions](https://docs.openvino.ai/2023.3/openvino_docs_install_guides_installing_openvino_from_archive_linux.html) and then I configured an additional logical link to:
```console
$ pushd /opt/intel/
$ sudo ln -s openvino_2023.3.0 openvino
```

### Building
Before starting I needed to make sure that by GPU can be detected by the
OpenVINO or I'd get the following compilation error:
```console
/home/danielbevenius/work/ai/openvino/openvino/src/plugins/intel_gpu/src/plugin/plugin.cpp:393:24: error: the mangled name of ‘ov::intel_gpu::<lambda(std::string, std::string, bool)>::operator std::__cxx11::basic_string<char> (*)(std::string, std::string, bool)() const’ changed between ‘-fabi-version=11’ (‘_ZNK2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_bE_cvPFS6_S6_S6_bEEv’) and ‘-fabi-version=18’ (‘_ZNK2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_bE_cvPFS7_S7_S7_bEEv’) [-Werror=abi]
  393 | auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
      |                        ^
/home/danielbevenius/work/ai/openvino/openvino/src/plugins/intel_gpu/src/plugin/plugin.cpp:393:24: error: the mangled name of ‘static std::__cxx11::basic_string<char> ov::intel_gpu::<lambda(std::string, std::string, bool)>::_FUN(std::string, std::string, bool)’ changed between ‘-fabi-version=11’ (‘_ZN2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_bE_4_FUNES6_S6_b’) and ‘-fabi-version=18’ (‘_ZN2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_bE_4_FUNES7_S7_b’) [-Werror=abi]
/home/danielbevenius/work/ai/openvino/openvino/src/plugins/intel_gpu/src/plugin/plugin.cpp:393:24: error: the mangled name of ‘ov::intel_gpu::<lambda(std::string, std::string, bool)>’ changed between ‘-fabi-version=11’ (‘_ZNK2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_bE_clES6_S6_b’) and ‘-fabi-version=18’ (‘_ZNK2ov9intel_gpu15StringRightTrimMUlNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES7_bE_clES7_S7_b’) [-Werror=abi]
/home/danielbevenius/work/ai/openvino/openvino/src/plugins/intel_gpu/src/plugin/plugin.cpp: In lambda function:
/home/danielbevenius/work/ai/openvino/openvino/src/plugins/intel_gpu/src/plugin/plugin.cpp:266:9: error: the mangled name of ‘ov::intel_gpu::Plugin::query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const::<lambda(std::shared_ptr<ov::Node>)>’ changed between ‘-fabi-version=11’ (‘_ZZNK2ov9intel_gpu6Plugin11query_modelERKSt10shared_ptrIKNS_5ModelEERKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_3AnyESt4lessISE_ESaISt4pairIKSE_SF_EEEENKUlS2_INS_4NodeEEE0_clESQ_’) and ‘-fabi-version=18’ (‘_ZZNK2ov9intel_gpu6Plugin11query_modelERKSt10shared_ptrIKNS_5ModelEERKSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_3AnyESt4lessISE_ESaISt4pairIKSE_SF_EEEENKUlS2_INS_4NodeEEE_clESQ_’) [-Werror=abi]
  266 |         [&prog](std::shared_ptr<ov::Node> node) {
      |         ^
```

So first source the following file:
```console
$ source fundamentals/llama.cpp/cuda-env.sh
```

Then I could build the OpenVINO toolkit:
```console
$ git clone https://github.com/openvinotoolkit/openvino.git
$ git submodule update --init --recursive
$ sudo ./install_build_dependencies.sh
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release .. -DENABLE_INTEL_GPU=OFF
$ cmake --build . --parallel 8
$ cmake --install . --prefix ~/work/ai/openvino/openvino_dist
```
One thing to note is the reaon I wanted to build OpenVINO is that it is the
default backend available for wasi-nn in wasmtime. And there is a specific
version that is used with wasmtime wasi-nn. Just building the latest and
sourcing the OpenVINO setupvars.sh file will not work. I think this might be
because the OpenVINO version used in openvino 0.6.0 is 2022.1.0 and it does
not use the newer OpenVINO 2.0 API. There is an open issue for this:
https://github.com/intel/openvino-rs/issues/53

But trying to build an older version of OpenVINO does not seem to work and I
spent way too much time trying to get it to work. Having to dymanically link
with external libraries it very painful and this is a very good example of this.


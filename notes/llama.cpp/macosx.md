## Macos notes for llama.cpp

### Building with xcodebuild
We need to install an iOS platform install to build which can be installed using:
```console
$ xcodebuild -downloadPlatform iOS
```

To build the project with xcodebuild, use the following command:
```console
$ xcodebuild -scheme llama -destination "generic/platform=iOS"
```

### multiple resources named 'ggml-metal.metal' in target 'llama'
I ran into this issue when trying to build the project with xcodebuild. The error message is:
```console
$ xcodebuild -scheme llama -destination "generic/platform=iOS"
Command line invocation:
    /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -scheme llama -destination generic/platform=iOS

User defaults from command line:
    IDEPackageSupportUseBuiltinSCM = YES

Resolve Package Graph
multiple resources named 'ggml-metal.metal' in target 'llama'multiple resources named 'ggml-metal-embed.metal' in target 'llama'

Resolved source packages:
  llama: /Users/danbev/work/llama.cpp

2024-11-01 07:02:29.107 xcodebuild[45972:9437010] Writing error result bundle to /var/folders/7h/g216wj3x0qldxw27twph8mwr0000gn/T/ResultBundle_2024-01-11_07-02-0029.xcresult
xcodebuild: error: Could not resolve package dependencies:
  multiple resources named 'ggml-metal.metal' in target 'llama'
  multiple resources named 'ggml-metal-embed.metal' in target 'llama'
```
This was due to there being a `ggml-metal.metal` file in the build directory:
```console
$ find . -name ggml-metal.metal
./build/bin/ggml-metal.metal
./ggml/src/ggml-metal.metal
```
Removing this file allowed the build to progress.

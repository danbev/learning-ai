## Wasm for ML
The trained models are often deployed to various devices and using wasm makes
them portable.
The usage of WASI is required as ML usually requires special hardware like GPUs
and TPUs (also AVX and SSE for CPUs) and WASI allows the ML code to access that
hardware.


wasi-nn is focused on inference and not training, at least at this stage.

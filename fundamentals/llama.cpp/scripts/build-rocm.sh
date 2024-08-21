#cmake -B ./build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_HIPBLAS=ON -DCMAKE_C_COMPILER=/opt/rocm-6.1.2/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-6.1.2/llvm/bin/clang++ -DAMDGPU_TARGETS='gfx1030;gfx1100;gfx1101;gfx1102'

cmake -B ./build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_HIPBLAS=ON -DCMAKE_C_COMPILER=/opt/rocm-6.1.2/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-6.1.2/llvm/bin/clang++ -DAMDGPU_TARGETS='gfx1030'

cmake --build ./build --config Release -- -j8

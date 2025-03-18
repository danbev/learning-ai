## Whisper windows-cublas CI build issue
Currently the CI job that build Whisper on Windows using CUDA (cublas, CUDA BLAS) takes
a long time, more than 3 hours to complete. This task should invesitgate this issue and
find a solution to reduce the build time.

In naively tried to add ccache to improve the build time, but it did not work, at least
not as much as I hoped for (it is still building so I don't have exact numbers at this
point.

### Job definition
```yaml
  windows-cublas:
    runs-on: windows-2019
    strategy:
      matrix:
        build: [Release]
        arch: [x64]
        cublas: [ON]
        sdl2: [ON]
        cuda-toolkit: [12.2.0, 11.8.0]
        include:
          - arch: x64
            sdl2: ON
            sdl2_ver: 2.28.5
    steps:
      - name: Clone repository
        uses: actions/checkout@v4

      - name: Install ccache
        uses: hendrikmuhs/ccache-action@v1.2.16
        with:
          key: ${{ github.job }}-${{ matrix.cuda-toolkit }}-${{ matrix.build }}
          variant: sccache
          evict-old-files: 1d

      - name: Add msbuild to PATH
        uses: microsoft/setup-msbuild@v2

      - name: Install CUDA Toolkit
        id: cuda-toolkit
        uses: Jimver/cuda-toolkit@v0.2.15
        with:
          cuda: '${{ matrix.cuda-toolkit }}'

      - name: Install 7-Zip
        run: choco install 7zip -y

      - name: Fetch SDL2 and set SDL2_DIR
        if: matrix.sdl2 == 'ON'
        run: |
          Invoke-WebRequest -Uri https://github.com/libsdl-org/SDL/releases/download/release-${{ matrix.sdl2_ver }}/SDL2-devel-${{ matrix.sdl2_ver }}-VC.zip -OutFile sdl2.zip
          7z x sdl2.zip
          echo "SDL2_DIR=${{ github.workspace }}\SDL2-${{ matrix.sdl2_ver }}\cmake" | Out-File -FilePath $env:GITHUB_ENV -Append
          echo "${{ github.workspace }}\SDL2-${{ matrix.sdl2_ver }}\cmake" > SDL2_PATH.txt

      - name: Configure CMake
        shell: cmd
        run: |
          cmake -S . -B ./build -A ${{ matrix.arch }} ^
            -DCMAKE_BUILD_TYPE=${{ matrix.build }} ^
            -DGGML_CUDA=${{ matrix.cublas }} ^
            -DCMAKE_CUDA_ARCHITECTURES=all  ^
            -DWHISPER_SDL2=${{ matrix.sdl2 }} ^
            -DSDL2_DIR="%SDL2_DIR%"

      - name: Build Project
        shell: cmd
        run: |
          cd ./build
          cmake --build . --config ${{ matrix.build }}

      - name: Copy CUDA DLLs
        run: |
          Get-ChildItem "${{ steps.cuda-toolkit.outputs.CUDA_PATH }}/bin/" -Filter "*.dll" |
          Copy-Item -Destination "build/bin/${{ matrix.build }}"

      - name: Copy SDL2.dll
        if: matrix.sdl2 == 'ON'
        run: copy "$env:SDL2_DIR/../lib/${{ matrix.arch }}/SDL2.dll" build/bin/${{ matrix.build }}

      - name: Upload binaries
        uses: actions/upload-artifact@v4
        with:
          name: whisper-cublas-${{ matrix.cuda-toolkit }}-bin-${{ matrix.arch }}
          path: build/bin/${{ matrix.build }}
```
So just to be clear on this will create two two jobs created by this single definition:

* Release, x64, cublas, sdl2, cuda-toolkit-12.2.0
* Release, x64, cublas, sdl2, cuda-toolkit-11.8.0

Sdl2 is a library that is used by Whisper for rendering the spectrogram (I think).

### Build log
The NVIDIA CUDA compiler `nvcc` is taking a long time to compile the CUDA source files.
Here is an example of the output:
```console
 D:\a\whisper.cpp\whisper.cpp\build\ggml\src\ggml-cuda>"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"  --use-local-env -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64" -x cu   -I"D:\a\whisper.cpp\whisper.cpp\ggml\src\ggml-cuda\.." -ID:\a\whisper.cpp\whisper.cpp\ggml\src\..\include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include"     --keep-dir x64\Release -use_fast_math -maxrregcount=0   --machine 64 --compile -cudart static -std=c++17 -arch=all -Xcompiler="/EHsc -Ob2"   -D_WINDOWS -DNDEBUG -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -D_XOPEN_SOURCE=600 -D_CRT_SECURE_NO_WARNINGS -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_SHARED -D"CMAKE_INTDIR=\"Release\"" -Dggml_cuda_EXPORTS -D_WINDLL -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Dggml_cuda_EXPORTS -Xcompiler "/EHsc /W1 /nologo /O2 /FS   /MD /GR" -Xcompiler "/Fdggm
  count-equal.cu
  tmpxft_00001954_00000000-7_count-equal.compute_90.cudafe1.cpp
  Compiling CUDA source file ..\..\..\..\ggml\src\ggml-cuda\cpy.cu...
  ```

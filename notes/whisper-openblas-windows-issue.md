## OpenBLAS Windows Issue
This is an issue that was reported by SubtitleEdit where after upgrading to
whisper.cpp 1.7.2/1.7.3, OpenBLAS was significantly slower than the previous version.

Issue: https://github.com/ggml-org/whisper.cpp/issues/2666

In whisper.cpp openblas is build liks this:
https://github.com/ggml-org/whisper.cpp/actions/runs/14221101401/job/39849066067
```console
Run vcpkg install --triplet=x64-windows openblas
  vcpkg install --triplet=x64-windows openblas
  choco install pkgconfiglite
  shell: C:\Program Files\PowerShell\7\pwsh.EXE -command ". '{0}'"
  env:
    BRANCH_NAME: master
    ubuntu_image: ubuntu:22.04
    VCPKG_BINARY_SOURCES: clear;x-gha,readwrite
    ACTIONS_CACHE_URL: https://acghubeus2.actions.githubusercontent.com/hrUrPcnOhqcq0zA2A9mVtpcyra8dMaKFFvLxk1wmrqGLNBXfHg/
    ACTIONS_RUNTIME_TOKEN: ***
Computing installation plan...
The following packages will be built and installed:
    openblas:x64-windows@0.3.29
```

And these are the CMake options used:
```console
cmake -S . -B ./build -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"  ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DWHISPER_SDL2=ON
```

There is also copy task for the openblas.dll:
```console
copy "C:/vcpkg/packages/openblas_x64-windows/bin/openblas.dll" build/bin/Release
```
I'm not sure if this matters or not but when I build on windows with the same options
I will get an `openblas.dll` in the `build/bin/Release` directory:	
```console
> ls .\build\bin\Release\openblas.dll


    Directory: C:\Users\danie\work\ai\whisper.cpp\build\bin\Release


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         5/15/2025   4:21 PM        1756160 openblas.dll
```

I'm trying to understand if this could be a potential issue not using the
same `openblas.dll`. I'll try to see if I see a difference when using the
one from the openblas package directly. The are identical so I don't think
this is the issue.		


So the version that I have is the same `0.3.29`:
```console
> ./vcpkg list openblas
openblas:x64-windows                              0.3.29              OpenBLAS is an optimized BLAS library based on G...```
```
This is also the lastest version of OpenBLAS:
https://github.com/OpenMathLib/OpenBLAS/release://github.com/OpenMathLib/OpenBLAS/releases


Now in SubtitleEdit they have the following in `src\SubtitleEdit\Forms\AudioToText\WhisperDownload.cs`:
```csharp
public sealed partial class WhisperDownload : Form
{
    private const string DownloadUrl64Cpp = "https://github.com/SubtitleEdit/support-files/releases/download/whispercpp-175/whisper-blas-bin-x64.zip";
    private const string DownloadUrl32Cpp = "https://github.com/SubtitleEdit/support-files/releases/download/whispercpp-175/whisper-blas-bin-Win32.zip";
    private readonly CancellationTokenSource _cancellationTokenSource;
    private readonly string _whisperChoice;
    private string _tempFileName;

    private static readonly string[] Sha512HashesCpp =
    {
        "055f632cdb2da4ab1c04bc9d0a7ecf76b43999eed66e99d327064dea960ba7105728556c2d276f9703d3940587a38447fe9dbb546372cfbab9b769b790def113", // v175/whisper-blas-bin-x64.zip
        "6c54435370961650d1896eac6b66b2fa42631cf4d27c22d9da2e425c513df7882d7f824290e93397c4cf52fbe96765da913cb14c70b26ff599e0472afcae2774", // v175/whisper-blas-bin-Win32.zip
    };

    private static readonly string[] OldSha512HashesCpp =
    {
        "db15572e89c034754022e7542637b2e57f1e8b54484f05fa842bddf943cf41e95bb43a305f30c7a9f1cdd164ba4deca6178eadcb513861f3d470a508b9b385cc", // v173/whisper-blas-bin-x64.zip
        "a6770e8d9d90a2c39f74f53bda06a79012738d4ea9b83c53377776dd0e360b0634464447a87197303cea104ba5412b320d1b4742a9cd1de1c7d888091a65b189", // v173/whisper-blas-bin-Win32.zip
        "66c97cff5514827c1f9496d69df7a86ba786dc94b6ba78350ba5e43ce28af5267de9b859232243c2dfd020fb33a7a8c80739b9169fdbaad330cde417e4afff08", // v172/whisper-blas-bin-x64.zip
        "3833976a5278deac3983050944723963b5a2e1721734376bc25d64cc3ec87ff912dc3b12842d2e11ca85ce923b0a54f0c77d71f161fe97627bdc33ee6dcf64e2", // v172/whisper-blas-bin-Win32.zip
        "4da4e0c5ed15063ea784a1be9d3f1b791c4c380d9f536fb120b93c3d0cbed853e74f318c394480b620156ad73704d4d51bec82badac7ae52929e0a2d53cd5e1d", // v1.5.4/whisper-blas-clblast-bin-x64.zip
        "1fdcd4c57f19d09507f63c5a4ac4de93858433517c549c6f8b5fe3508f4e35d2535638e59fcddc9f6d36df3d87630870a12c87c80b442da3de6ddeaf859ef0c6", // v1.5.4/whisper-blas-bin-Win32.zip"
        "179fd78ce1691ab885118d0359cfaeb753a7aaae61d378cc47ea0c215fe01c929259858a081eff47ef2c2c07a3b5f6d0f30b894ce246aab0d6083ccc6fd517ab", // v1.5.3/whisper-blas-clblast-bin-x64.zip
        "cf0ebadb964701beb01ec1ac98eb4a85dae03e356609d65b2f7fb4f8b12aee00609369bfd4e1a40930eaeb95f3e0d208535a967dc6f4af03a564ae01f654d364", // v1.5.3/whisper-blas-bin-Win32.zip
        "1667a86007a6f6d36a94fae0c315c3321eb2572274be8ac540d141be198993d306554aabce1b5f34ac10ffdae09b4c227efba8a4f16978addd82836dc2156c34", // v1.5.2/whisper-blas-bin-x64.zip
        "647c727417bc6a7c90c7460100214426fc1b82fee1ce9924eaec71b46466920b1045c1a534a72782a0d6dcc31541a85a5ad62bfb635c815738c95fafea368cd4", // v1.5.2/whisper-blas-bin-Win32.zip
        "4dad22644af9770ecd05f1959adbe516e0948fb717d0bc33d5f987513f619162159aa2092b54a535e909846caca8dbf53f34c9060dadb43fc57b2c28e645dd73", // v1.5.1/whisper-blas-bin-x64.zip
        "00af057d6ba4005ac1758a713bbe21091796202a81ec6b7dcce5cd9e7680734730c430b75b88a6834b76378747bc4edbcf14a4ed7429b07ea9a394754f4e3368", // v1.5.1/whisper-blas-bin-Win32.zip
        "102cd250958c3158b96453284f101fadebbb0484762c78145309f7d7499aa2b9c9e01e5926a634bd423aee8701f65c7d851a19cb5468364697e624a2c53a325d", // v1.5.0/whisper-blas-bin-x64.zip
        "0bc8df7ca4fdd32a80a9f8e7568b8668221a4205ff8fc3d04963081c14677c6a97e4510e7bb12d7b110fc9a88553aeaa53eff558262ff2d725cef52b3100b149", // v1.5.0/whisper-blas-bin-Win32.zip
        "fc1878c3b7200d0531c376bbe52319a55575e3ceeeacecbee54a366116c30eb1aa3d0a34c742f9fd5a47ffb9f24cba75653d1498e95e4f6f86c00f6d5e593d2a", // v1.4.0/whisper-blas-bin-x64.zip
        "44cb0f326ece26c1b41bd0b20663bc946990a7c3b56150966eebefb783496089289b6002ce93d08f1862bf6600e9912ac62057c268698672397192c55eeb30a2", // v1.4.0/whisper-blas-bin-Win32.zip
        "e193e845380676b59deade82d3f1de70ac54da3b5ffa70b4eabb3a2a96ad312d8f197403604805cef182c85c3a3370dd74a2b0b7bccf2d95b1022c10ce8c7b79", // 64-bit OpenBLAS
        "4218423f79d856096cdc8d88aad2e361740940e706e0b1d07dc3455571022419ad14cfef717f63e8fc61a7a1ef67b6722cec8b3c4c25ad7f087a23b1b89c5d91", // 32-bit OpenBLAS
        "a6a75a5d63b933c3529a500b7dd8b330530894b09461bb0a715dbedb31bf2e3493238e86af6d7cc64f3af196a6d61d96bb23853f98d21c8172d5d53d7aad33d9", // 64-bit OpenBLAS
        "92f64f207c400c7c0f1fc27006bf2a1e4170fdc63d045dfdf0a0848b3d727f2763eccfb55e10b6e745e9d39892d24cb9b4c471594011d041458c1ff8722e1ffc", // 32-bit OpenBLAS
        "f2073d5ce928e59f7717a82f0330e4d628c81e6cb421b934b4792ac16fb9a33fb9482812874f39d4c7ca02a47a9739d5dd46ddc2e0abc0eb1097dc60bb0616b2", // AVX2 64-bit
        "0d65839e3b05274b3edeef7bbb123c9a6ba04b89729011b758210a80f74563a1e58ca7da0a25e63e42229a2d3dd57a2cb6ce993474b13381871f343b75c3140f", // SSE2 64-bit
        "9f9ce1b39610109bc597b296cb4c1573fa61d33eeaef2a38af44bb2d696fa7c1da297520630ada2470d740edb18a17fe3cca922ad12a307476e27862906450e5", // AVX2 32-bit
        "aab8e7349a7051fb35f2294da3c4993731f47ce2d45ba4c6d4b2b106d0e3236a0082b68e67eb612fec1540e60ae9994183bd41f7fee31c23ec192cbd4155e3c2", // SSE2 32-bit
        "b69bd16bd4d11191b7b1980157d09cb1e489c804219cd336cd0b58182d357b5fff491b29ab8796d1991a9c8f6c8537f475592400b7f4e1244fdfdda8c970a80c", // AVX2 64-bit
        "8e45e147397b688e0ff814f6ef87fd6703748a4f9170fa6498b9428db47bbf9140c7479d016b8e201340ac6627e3f9632c70aa36e7a883355b9abf30e6796ae9", // SSE2 64-bit
        "87799433a5a29b3beaa5a58dfc22471e2c1eb1c9821b6a337b40d8e3c1b4dae9118e10d3a278664fe0f36ba6543ac554108593045640f62711b95f4c2a113b74", // SSE2 32-bit
        "58834559f7930c8c3dff6e20963cb86a89ca0228752d35b2486f907e59435a9adc5f5fb13c644f66bedea9ce0368c193d9632366719d344abbd3c0eb547e7110", // SSE2 64-bit
        "999863541ffbbce142df271c7577f31fef3f386e3ccbb0c436cb21bb13c7557a28602a2f2c25d6a32a6bca7953a21e086a4af3fbdc84b295e994d3452d3af5bc",
        "3c8a360d1e097d229500f3ccdd66a6dc30600fd8ea1b46405ed2ec03bb0b1c26c72cac983440b5793d24d6983c3d76482a17673251dd89f0a894a54a6d42d169", // AVX2 64-bit
        "96f8e6c073afc75062d230200c9406c38551d8ce65f609e433b35fb204dc297b415eb01181adb6b1087436ae82c4e582b20e97f6b204acf446189bde157187b7", // AVX2 32-bit
        "2a9e10f746a1ebe05dffa86e9f66cd20848faa6e849f3300c2281051c1a17b0fc35c60dc435f07f5974aa1191000aaf2866a4f03a5fe35ecffd4ae0919778e63", // SSE2 32-bit
        "2f6ab662aecd09ad5d06690ad01981d155630462da077072301b624efed702559616bf368a640864d44d8f50927d56d252345117084ef6e795b67964f6303fe4", // v171/whisper-blas-bin-x64.zip
        "3ff1490ca2c0fa829ab0c4e5b485c4e93ed253adc30557ef4f369920925a246ffc459c122e0f3c0b166ef94b50e7f1e7e48c29c43ca551c3d8905f5ef3d8004c", // v171/whisper-blas-bin-Win32.zip
    };


    private const string DownloadUrl64CppCuBlas = "https://github.com/SubtitleEdit/support-files/releases/download/whispercpp-175/whisper-cublas-12.2.0-bin-x64.7z";

    private static readonly string[] Sha512HashesCppCuBlas =
    {
        "f901750abab46791ba91a3a6575f67f368fa01a903028bb35c3f6e347a6d0fd3b36dc6c4b81644250ec12c4ff8e7a4dff605eb87d14aedc3908e1335d8a34194", // v175/whisper-cublas-12.2.0-bin-x64.zip
        "1b105d38702a01ab8e98b31a690040ca54861b5e55773fff9242f33ba7b0718a6e9f25231ed107a7db0292d8349d508b18300f6c95a8e4234faef27cb05887aa", // v172/whisper-cublas-12.2.0-bin-x64.zip
        "37c77ce10739b67588fdc1ca06ac8ff3c578d230974af6c5d90cf80f0d85af1a28f6827447b7b63699c21a5fddfeedeb3bd6cf8a64dd859598e94faef2b9ba3e", // v171/whisper-cublas-12.2.0-bin-x64.zip
        "e0279cfc73473b3a9530f44906453c34d9d197cb1cdec860447ce226dd757cc13e3f5f2a22386b95553fc99961e56baf92b20ac1be4217c6a60e749bb5e95cc0", // v1.5.3/whisper-cublas-12.2.0-bin-x64.zip
        "9ca711e835075249a7ecbeb6188be2da2407f94ca04740ba56b984601e68df994e607f03c3816617d92562ed3820b170c48ec82840880efd524da6dfe5b70691", // v1.5.4/whisper-cublas-12.2.0-bin-x64.zip
        "9ca711e835075249a7ecbeb6188be2da2407f94ca04740ba56b984601e68df994e607f03c3816617d92562ed3820b170c48ec82840880efd524da6dfe5b70691", // v1.6.0/whisper-cublas-12.2.0-bin-x64.zip
    };
```
So these files are downloaded from https://github.com/SubtitleEdit/support-files. Looking there we can find tags for the version and
download and inspect them. I'm not sure how these are built though as there are now actions in that repo.


Lets download version 1.7.5:
```console
Invoke-WebRequest "https://github.com/SubtitleEdit/support-files/releases/download/whispercpp-175/whisper-blas-bin-x64.zip" -OutFile "whisper-175.zip"
Expand-Archive "whisper-175.zip" -DestinationPath "whisper-175"
```
And also version 1.7.1:
```console
Invoke-WebRequest "https://github.com/SubtitleEdit/support-files/releases/download/whispercpp-171/whisper-blas-bin-x64.zip" -OutFile "whisper-171.zip"
Expand-Archive "whisper-171.zip" -DestinationPath "whisper-171"
```
```console
> ls .\whisper-175\openblas.dll


    Directory: C:\Users\danie\work\ai\whisper.cpp\openblas-example-dir\whisper-175


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----          4/2/2025   6:41 AM        1756160 openblas.dll


> ls .\whisper-171\libopenblas.dll


    Directory: C:\Users\danie\work\ai\whisper.cpp\openblas-example-dir\whisper-171


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         10/7/2024  10:32 AM       50860828 libopenblas.dll
```
Notice the difference in size. About 48MB vs 1.7MB and and probably means that there
are missing CPU specific kernels (AVX, AVX2, etc) and perhaps this explains the
slowdown in performance.

The version of Subtitle Edit 4.0.8 is using Whisper CPP 1.7.1 and if we look at how
that version's [windows-blas](https://github.com/ggml-org/whisper.cpp/blob/v1.7.1/.github/workflows/build.yml)
github action job looks like we find:
```
 windows-blas:
    runs-on: windows-latest

    strategy:
      matrix:
        build: [Release]
        arch: [Win32, x64]
        blas: [ON]
        sdl2: [ON]
        include:
          - arch: Win32
            obzip: https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.25/OpenBLAS-0.3.25-x86.zip
            s2arc: x86
          - arch: x64
            obzip: https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.25/OpenBLAS-0.3.25-x64.zip
            s2arc: x64
          - sdl2: ON
            s2ver: 2.28.5
    
    ...
    steps:
      - name: Clone
        uses: actions/checkout@v4

      - name: Add msbuild to PATH
        uses: microsoft/setup-msbuild@v2

      - name: Fetch OpenBLAS
        if: matrix.blas == 'ON'
        run: |
          C:/msys64/usr/bin/wget.exe -qO blas.zip ${{ matrix.obzip }}
          7z x blas.zip -oblas -y
          copy blas/include/cblas.h .
          copy blas/include/openblas_config.h .
          echo "OPENBLAS_PATH=$env:GITHUB_WORKSPACE/blas" >> $env:GITHUB_ENV
```

```console
Invoke-WebRequest "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.25/OpenBLAS-0.3.25-x64.zip" -OutFile "OpenBLAS-0.3.25.zip"
Expand-Archive "OpenBLAS-0.3.25.zip" -DestinationPath "OpenBLAS-0.3.25"
```

And lets download the latest version too:
```console
Invoke-WebRequest "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64_64.zip" -OutFile "OpenBLAS-0.3.29.zip"
Expand-Archive "OpenBLAS-0.3.29.zip" -DestinationPath "OpenBLAS-0.3.29"
```

### Summary
So the `libopenblas.dll` contains multiple optimized kernels for different CPU
arcitectures. For example:
```console
libopenblas.dll (51MB) contains:
├── Generic kernels (SSE2) - baseline for any x64 CPU
├── AVX kernels - for newer CPUs (2011+)
├── AVX2 kernels - for even newer CPUs (2013+)
├── FMA kernels - for CPUs with Fused Multiply-Add
├── AVX-512 kernels - for latest CPUs (2016+)
└── Runtime CPU detection logic
```
At runtime OpenBLAS will detect the CPU and use the most optimized kernel for it.
The name `libopenblas` is the official OpenBLAS library name built with full
configuration and opts.

And `openblas.dll` is often used by package managers like vcpkg. These may be
stripped down versions.

### Suggested solution
So I think this issue could be solved by building similar to something like
this:
```console
set OPENBLAS_PATH=%cd%\OpenBLAS-0.3.29

cmake -S . -B build -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DBLAS_LIBRARIES="%OPENBLAS_PATH%/lib/libopenblas.lib" ^
    -DBLAS_INCLUDE_DIRS="%OPENBLAS_PATH%/include" ^
    -DWHISPER_SDL2=ON ^
    -DSDL2_DIR="C:\Users\danie\work\audio\SDL2-devel-2.32.4-VC\SDL2-2.32.4\cmake"

cmake --build build --config Release

copy "%OPENBLAS_PATH%\bin\libopenblas.dll" ".\build\bin\Release"
```

This change would have to be made with the SubtitleEdit build system that
produces the whisper.cpp artifacts and uploads them to
https://github.com/SubtitleEdit/support-files.



### Test

The following is run built with `libopenblas.dll` version `0.3.29`:
```console
> .\whisper-cli.exe -m ..\..\ggml\
ALL_BUILD.vcxproj          cmake_install.cmake        ggml.sln                   ggml-version.cmake         INSTALL.vcxproj.filters    RUN_TESTS.vcxproj.filters
ALL_BUILD.vcxproj.filters  CMakeFiles                 ggml-config.cmake          INSTALL.vcxproj            RUN_TESTS.vcxproj          src
> .\whisper-cli.exe -m ..\..\..\models\ggml-base.en.bin -f ..\..\..\samples\
.gitignore          duty.ogg            Euskal_Dantzak.ogg  gb0.ogg             jfk.mp3             jfk.wav             jfk.wav.vtt         README.md
> .\whisper-cli.exe -m ..\..\..\models\ggml-base.en.bin -f ..\..\..\samples\duty.ogg
whisper_init_from_file_with_params_no_state: loading model from '..\..\..\models\ggml-base.en.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0
whisper_init_with_params_no_state: devices    = 2
whisper_init_with_params_no_state: backends   = 2
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 2 (base)
whisper_model_load: adding 1607 extra tokens
whisper_model_load: n_langs       = 99
whisper_model_load:          CPU total size =   147.37 MB
whisper_model_load: model size    =  147.37 MB
whisper_backend_init_gpu: no GPU found
whisper_backend_init: using BLAS backend
whisper_init_state: kv self size  =    6.29 MB
whisper_init_state: kv cross size =   18.87 MB
whisper_init_state: kv pad  size  =    3.15 MB
whisper_init_state: compute buffer (conv)   =   16.26 MB
whisper_init_state: compute buffer (encode) =   85.86 MB
whisper_init_state: compute buffer (cross)  =    4.65 MB
whisper_init_state: compute buffer (decode) =   96.35 MB

system_info: n_threads = 4 / 16 | WHISPER : COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: processing '..\..\..\samples\duty.ogg' (4094643 samples, 255.9 sec), 4 threads, 1 processors, 5 beams + best of 5, lang = en, task = transcribe, timestamps = 1 ...


[00:00:00.000 --> 00:00:09.500]   "The first duty of a government is to be true to itself.
[00:00:09.500 --> 00:00:14.000]   This does not mean perfection, it means it plans a strive for perfection.
[00:00:14.000 --> 00:00:16.500]   It means loyal to issue ideals.
[00:00:16.500 --> 00:00:22.500]   The ideals of America were set out in the Declaration of Independence and adopted in the Constitution.
[00:00:22.500 --> 00:00:27.500]   They did not represent perfection, attain, but perfection firing.
[00:00:27.500 --> 00:00:29.500]   The fundamental principle was freedom.
[00:00:29.500 --> 00:00:33.000]   The fathers knew that this was not yet apprehended.
[00:00:33.000 --> 00:00:39.000]   They formed a government firm in the faith that it was ever to press toward this high march.
[00:00:39.000 --> 00:00:44.000]   In selfishness, in greed, in lust, for gain, it turned aside.
[00:00:44.000 --> 00:00:47.000]   Inflaving others, it became itself in place.
[00:00:47.000 --> 00:00:51.500]   Bondage in one part consumes freedom in all parts.
[00:00:51.500 --> 00:00:56.500]   The government of the fathers, teaching to be true to itself, was pericking.
[00:00:56.500 --> 00:01:04.500]   Five score and ten years ago, that divine providence, which infinite reputation has made only the more a miracle,
[00:01:04.500 --> 00:01:08.500]   sent into the world a new life, destined to save a nation.
[00:01:08.500 --> 00:01:12.000]   No star, no sign foretold his coming.
[00:01:12.000 --> 00:01:19.500]   About his cradle hall was poor and mean, save only the source of all great men, the love of a wonderful woman.
[00:01:19.500 --> 00:01:24.500]   When she faded away in his tender years from her death bed in humble poverty,
[00:01:24.500 --> 00:01:27.000]   she dared her son with greatly.
[00:01:27.000 --> 00:01:32.000]   There can be no proper observance of a birthday which forgets the mother.
[00:01:32.000 --> 00:01:36.500]   Into his origins, as into his life, men long have looked and wondered.
[00:01:36.500 --> 00:01:44.000]   In wisdom, great, but in humility, greater, in justice, strong, but in compassion, stronger.
[00:01:44.000 --> 00:01:48.500]   He became a leader of men by being a follower of the truth.
[00:01:48.500 --> 00:01:53.000]   He overcame evil with good, his presence filled the nation.
[00:01:53.000 --> 00:01:57.500]   He broke the might of a fraction. He restored a race to its birthright.
[00:01:57.500 --> 00:02:07.000]   His mortal frame was vanished, but his spirit increases with increasing years, the richest legacy of the greatest century.
[00:02:07.000 --> 00:02:10.500]   Men show by what they worship, what they are.
[00:02:10.500 --> 00:02:18.000]   It is no accident that before the great example of American manhood, our people stand with respect and reverence.
[00:02:18.000 --> 00:02:24.500]   In Abraham Lincoln, he has revealed our ideal, the hope of our country fulfilled.
[00:02:24.500 --> 00:02:28.500]   He was the incarnation of what America was to be.
[00:02:28.500 --> 00:02:33.500]   Through him, the Almighty bestowed upon the nation a new birth of freedom.
[00:02:33.500 --> 00:02:38.500]   That this dear land of ours might be returned to the house of its fathers.
[00:02:38.500 --> 00:02:47.000]   We are the beneficiaries of a life of surpassing service, wise in wisdom and gentle in gentleness.
[00:02:47.000 --> 00:02:52.500]   Freedom has many sides and angles. Human slavery has been swept away.
[00:02:52.500 --> 00:02:57.000]   With security of personal rights has come security of property rights.
[00:02:57.000 --> 00:03:03.000]   The freedom of the human mind is recognized in the right of free speech and free press.
[00:03:03.000 --> 00:03:09.000]   The public schools have made education possible for all and eager and edgy trade.
[00:03:09.000 --> 00:03:16.000]   In political affairs, the vote of the humbly has long counted for as much as the vote of the most exalted.
[00:03:16.000 --> 00:03:23.500]   We are working towards the day when in our industrial life, equal honor shall fall to equal endeavor.
[00:03:23.500 --> 00:03:27.000]   Beauty is collective as well as personal.
[00:03:27.000 --> 00:03:31.000]   Law must rest on the eternal foundation of righteousness.
[00:03:31.000 --> 00:03:37.000]   Industry, truth, character cannot be conferred by acts or resolve.
[00:03:37.000 --> 00:03:46.000]   Government cannot relieve from time. Do the day's work. If it be to protect the rights of the weak, whoever is get, do it.
[00:03:46.000 --> 00:03:53.000]   If it be to help the powerful corporations better to serve the people, whatever the opposition, do that.
[00:03:53.000 --> 00:03:57.000]   Expect to be called a santata, but don't be a santata.
[00:03:57.000 --> 00:04:01.000]   Expect to be called a demagogue, but don't be a demagogue.
[00:04:01.000 --> 00:04:08.000]   We need a broader, grammar, beef of faith in the people. A faith that men desire to do right.
[00:04:08.000 --> 00:04:13.000]   That the government is founded upon a righteousness which will endure.

whisper_print_timings:     load time =   192.35 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =   116.01 ms
whisper_print_timings:   sample time =  2059.85 ms /  4325 runs (     0.48 ms per run)
whisper_print_timings:   encode time =  5274.02 ms /    10 runs (   527.40 ms per run)
whisper_print_timings:   decode time =    76.49 ms /    21 runs (     3.64 ms per run)
whisper_print_timings:   batchd time =  6026.30 ms /  4255 runs (     1.42 ms per run)
whisper_print_timings:   prompt time =  1406.18 ms /  1854 runs (     0.76 ms per run)
whisper_print_timings:    total time = 15338.52 ms
```

The following is run built with `openblas.dll` version `0.3.29` from vcpgk:
```console
set SDL2_PATH="C:\Users\danie\work\audio\SDL2-devel-2.32.4-VC\SDL2-2.32.4"
set VCPKG_PATH="C:\Users\danie\vcpkg"

cmake -S . -B build -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DCMAKE_TOOLCHAIN_FILE=%VCPKG_PATH%\scripts\buildsystems\vcpkg.cmake ^
    -DWHISPER_SDL2=ON ^
    -DSDL2_DIR="%SDL2_PATH%\cmake"

cmake --build build --config Release
```

```console
> .\whisper-cli.exe -m ..\..\..\models\ggml-base.en.bin -f ..\..\..\samples\duty.ogg
whisper_init_from_file_with_params_no_state: loading model from '..\..\..\models\ggml-base.en.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0
whisper_init_with_params_no_state: devices    = 2
whisper_init_with_params_no_state: backends   = 2
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 2 (base)
whisper_model_load: adding 1607 extra tokens
whisper_model_load: n_langs       = 99
whisper_model_load:          CPU total size =   147.37 MB
whisper_model_load: model size    =  147.37 MB
whisper_backend_init_gpu: no GPU found
whisper_backend_init: using BLAS backend
whisper_init_state: kv self size  =    6.29 MB
whisper_init_state: kv cross size =   18.87 MB
whisper_init_state: kv pad  size  =    3.15 MB
whisper_init_state: compute buffer (conv)   =   16.26 MB
whisper_init_state: compute buffer (encode) =   85.86 MB
whisper_init_state: compute buffer (cross)  =    4.65 MB
whisper_init_state: compute buffer (decode) =   96.35 MB

system_info: n_threads = 4 / 16 | WHISPER : COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: processing '..\..\..\samples\duty.ogg' (4094643 samples, 255.9 sec), 4 threads, 1 processors, 5 beams + best of 5, lang = en, task = transcribe, timestamps = 1 ...


[00:00:00.000 --> 00:00:09.500]   "The first duty of a government is to be true to itself.
[00:00:09.500 --> 00:00:14.000]   This does not mean perfection, it means it plans a strive for perfection.
[00:00:14.000 --> 00:00:16.500]   It means loyal to issue ideals.
[00:00:16.500 --> 00:00:22.500]   The ideals of America were set out in the Declaration of Independence and adopted in the Constitution.
[00:00:22.500 --> 00:00:27.500]   They did not represent perfection, attain, but perfection firing.
[00:00:27.500 --> 00:00:29.500]   The fundamental principle was freedom.
[00:00:29.500 --> 00:00:33.000]   The fathers knew that this was not yet apprehended.
[00:00:33.000 --> 00:00:39.000]   They formed a government firm in the faith that it was ever to press toward this high march.
[00:00:39.000 --> 00:00:44.000]   In selfishness, in greed, in lust, for gain, it turned aside.
[00:00:44.000 --> 00:00:47.000]   Inflaving others, it became itself in place.
[00:00:47.000 --> 00:00:51.500]   Bondage in one part consumes freedom in all parts.
[00:00:51.500 --> 00:00:56.500]   The government of the fathers, teaching to be true to itself, was pericking.
[00:00:56.500 --> 00:01:04.500]   Five score and ten years ago, that divine providence, which infinite reputation has made only the more a miracle,
[00:01:04.500 --> 00:01:08.500]   sent into the world a new life, destined to save a nation.
[00:01:08.500 --> 00:01:12.000]   No star, no sign foretold his coming.
[00:01:12.000 --> 00:01:19.500]   About his cradle hall was poor and mean, save only the source of all great men, the love of a wonderful woman.
[00:01:19.500 --> 00:01:24.500]   When she faded away in his tender years from her death bed in humble poverty,
[00:01:24.500 --> 00:01:27.000]   she dared her son with greatly.
[00:01:27.000 --> 00:01:32.000]   There can be no proper observance of a birthday which forgets the mother.
[00:01:32.000 --> 00:01:36.500]   Into his origins, as into his life, men long have looked and wondered.
[00:01:36.500 --> 00:01:44.000]   In wisdom, great, but in humility, greater, in justice, strong, but in compassion, stronger.
[00:01:44.000 --> 00:01:48.500]   He became a leader of men by being a follower of the truth.
[00:01:48.500 --> 00:01:53.000]   He overcame evil with good, his presence filled the nation.
[00:01:53.000 --> 00:01:57.500]   He broke the might of a fraction. He restored a race to its birthright.
[00:01:57.500 --> 00:02:07.000]   His mortal frame was vanished, but his spirit increases with increasing years, the richest legacy of the greatest century.
[00:02:07.000 --> 00:02:10.500]   Men show by what they worship, what they are.
[00:02:10.500 --> 00:02:18.000]   It is no accident that before the great example of American manhood, our people stand with respect and reverence.
[00:02:18.000 --> 00:02:24.500]   In Abraham Lincoln, he has revealed our ideal, the hope of our country fulfilled.
[00:02:24.500 --> 00:02:28.500]   He was the incarnation of what America was to be.
[00:02:28.500 --> 00:02:33.500]   Through him, the Almighty bestowed upon the nation a new birth of freedom.
[00:02:33.500 --> 00:02:38.500]   That this dear land of ours might be returned to the house of its fathers.
[00:02:38.500 --> 00:02:47.000]   We are the beneficiaries of a life of surpassing service, wise in wisdom and gentle in gentleness.
[00:02:47.000 --> 00:02:52.500]   Freedom has many sides and angles. Human slavery has been swept away.
[00:02:52.500 --> 00:02:57.000]   With security of personal rights has come security of property rights.
[00:02:57.000 --> 00:03:03.000]   The freedom of the human mind is recognized in the right of free speech and free press.
[00:03:03.000 --> 00:03:09.000]   The public schools have made education possible for all and eager and edgy trade.
[00:03:09.000 --> 00:03:16.000]   In political affairs, the vote of the humbly has long counted for as much as the vote of the most exalted.
[00:03:16.000 --> 00:03:23.500]   We are working towards the day when in our industrial life, equal honor shall fall to equal endeavor.
[00:03:23.500 --> 00:03:27.000]   Beauty is collective as well as personal.
[00:03:27.000 --> 00:03:31.000]   Law must rest on the eternal foundation of righteousness.
[00:03:31.000 --> 00:03:37.000]   Industry, truth, character cannot be conferred by acts or resolve.
[00:03:37.000 --> 00:03:46.000]   Government cannot relieve from time. Do the day's work. If it be to protect the rights of the weak, whoever is get, do it.
[00:03:46.000 --> 00:03:53.000]   If it be to help the powerful corporations better to serve the people, whatever the opposition, do that.
[00:03:53.000 --> 00:03:57.000]   Expect to be called a santata, but don't be a santata.
[00:03:57.000 --> 00:04:01.000]   Expect to be called a demagogue, but don't be a demagogue.
[00:04:01.000 --> 00:04:08.000]   We need a broader, grammar, beef of faith in the people. A faith that men desire to do right.
[00:04:08.000 --> 00:04:13.000]   That the government is founded upon a righteousness which will endure.

whisper_print_timings:     load time =   104.14 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =   114.92 ms
whisper_print_timings:   sample time =  2116.88 ms /  4325 runs (     0.49 ms per run)
whisper_print_timings:   encode time = 66273.03 ms /    10 runs (  6627.30 ms per run)
whisper_print_timings:   decode time =    74.25 ms /    21 runs (     3.54 ms per run)
whisper_print_timings:   batchd time =  6008.43 ms /  4255 runs (     1.41 ms per run)
whisper_print_timings:   prompt time = 16424.26 ms /  1854 runs (     8.86 ms per run)
whisper_print_timings:    total time = 91288.89 ms
```

Total times:
```console
whisper_print_timings:    total time = 15338.52 ms  (15.3s)
whisper_print_timings:    total time = 91288.89 ms  (91.3s)
```

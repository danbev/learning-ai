### FFmpeg parakeet.cpp integration
To try this out we need to first checkout the parakeet-support branch:
```console
$ git clone -b parakeet-support https://github.com/danbev/whisper.cpp.git
```
Then we build and install the library to a local directory named `build-install`:
```console
$ cat build-install.sh 
#!/bin/bash

set -e

build_dir=build
install_dir=build-install

rm -rf ${install_dir}
mkdir -p ${install_dir}

cmake -S . -B ${build_dir} -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/danbev/work/ai/whisper-work/${install_dir} \
    -DGGML_BACKEND_DIR=/home/danbev/work/ai/whisper-work/${install_dir}/lib \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_USE_CPU=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DWHISPER_ALL_WARNINGS=ON \
    -DWHISPER_FATAL_WARNINGS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89-real" \
    -DGGML_CPU_AARCH64=OFF \
    -DGGML_CUDA_F16=ON

cmake --build ${build_dir} -j 8
cmake --install ${build_dir} --prefix ${install_dir}
```

Then we need to check out the FFmpeg parakeet.cpp branch:
```console
$ git clone -b parakeet.cpp https://github.com/danbev/FFmpeg.git
```
And then build FFmpeg using the following configuration options and we explicitely
set `PKG_CONFIG_PATH` to point to the `pkgconfig` directory of the local
installation above:
```console
$ export PKG_CONFIG_PATH="/home/danbev/work/ai/whisper-work/build-install/lib/pkgconfig:$PKG_CONFIG_PATH"

$ ./configure --prefix=/usr --enable-version3 --disable-shared --enable-gpl \
  --enable-nonfree --enable-static --enable-pthreads --enable-filters \
  --enable-openssl --enable-runtime-cpudetect --enable-libvpx --enable-libx264 \
  --enable-libx265 --enable-libspeex --enable-libfreetype --enable-fontconfig \
  --enable-libzimg --enable-libvorbis --enable-libwebp --enable-libfribidi \
  --enable-libharfbuzz --enable-libass --enable-whisper --enable-parakeet

$ make
```
To run we need to set `LD_LIBRARY_PATH` to point to the `lib` directory of the local installation above so that the parakeet and whisper backends can be found at runtime. For macos this
would instead be `DYLD_LIBRARY_PATH`:
```console
$ export LD_LIBRARY_PATH=/home/danbev/work/ai/whisper-work/build-install/lib/:$LD_LIBRARY_PATH
```

After that it should be possible to run using the following command:
```console
$ ./ffmpeg -i gb1.wav -loglevel quiet -af parakeet=model=ggml-parakeet-tdt-0.6b-v3.bin:use_gpu=1:destination=- -f null -
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 11903 MiB):
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes, VRAM: 11903 MiB
load_backend: loaded CUDA backend from /home/danbev/work/ai/whisper-work/build-install/lib/libggml-cuda.so
load_backend: loaded CPU backend from /home/danbev/work/ai/whisper-work/build-install/lib/libggml-cpu-alderlake.so
My fellow Americans, this day has brought terrible news and great sadness to our country. At nine o'clock this morning, mission control in Houston lost contact with our space shuttle Columbia. A short time later, debris was seen falling from the skies above Texas. The Columbia's lost. There are no survivors. On board was a crew of seven Colonel Rick Husband, Lieutenant Colonel Michael Anderson, Commander Laurel Clark, Captain David Brown, Commander William McCool, Dr. Kulpna Shavla, and Ilan Ramon, a colonel in the Israeli Air Force. These men and women assumed great risk in the service to all humanity. In an age when spaceflight has come to seem almost routine, it is easy to overlook the dangers of travel by rocket and the difficulties of navigating the fierce outer atmosphere of the Earth. Because of their courage and daring and idealism, we will miss them all the more. All Americans today are thinking as well of the families of these men and women who have been given this sudden shock and grief. You're not alone. Our entire nation grieves with you, and those you love will always have the respect and gratitude of this country. The cause in which they died will continue. Mankind is led into the darkness beyond our world by the inspiration of discovery and the longing to understand. Our journey into space will go on. In the skies today, we saw destruction and tragedy. Yet farther than we can see, there is comfort and hope. In the words of the prophet Isaiah, lift your eyes and look to the heavens. Who created all these? He who brings out the starry hosts one by one and calls them each by name, because of his great power and mighty strength, not one of them is missing. The crew of the shuttle Columbia did not return safely to Earth. Yet we can pray that all are safely home. May God bless the grieving families, and may God continue to bless America.
```

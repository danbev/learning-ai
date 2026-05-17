## Hailo

### Installation
```console
$ sudo apt install libusb-1.0-0-dev
$ sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu -y
```

Clone the repository:
```console
$ pushd ..
$ git clone git@github.com:hailo-ai/hailort.git && cd hailort
$ cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release && cmake --build build --config release
$ popd
```

### Building
```console
$ make simple run-simple
g++ -std=c++17 -Wall -Wextra -I../hailort/hailort/libhailort//include src/simple.cpp -o simple -lhailort -L../hailort/build/hailort/libhailort/src
env LD_LIBRARY_PATH=../hailort/build/hailort/libhailort/src ./simple
HailoRT Version: 5.3.0
```

### Hardware
TODO: update when hardware arrives.



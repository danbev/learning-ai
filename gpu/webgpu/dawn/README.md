## Google Dawn example
This is a WebGPU example in C++ that used Google's Dawn library.


### Install Dawn
```console
$ git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
$ export PATH="${PWD}/depot_tools:${PATH}"
```
There is a `set-env.sh` script that can be used to set the environment
variable.

Next, clone the Dawn repository:
```console
$ git clone https://dawn.googlesource.com/dawn
$ cd dawn
```

Fetch dependencies using gclient:
```console
$ cp scripts/standalone.gclient .gclient
$ gclient sync
```

```console
$ sudo apt install libxinerama-dev libxcursor-dev libx11-xcb-dev
```
Build Dawn:
```console
$ mkdir -p out/Debug
$ cd out/Debug
$ cmake ../..
$ make -j8
```

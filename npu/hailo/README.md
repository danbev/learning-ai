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
* Raspberry Pi 5
* Raspberry Pi AI HAT+ 2 (Hailo-10H AI Accelerator chip)

Parts:
![image](../../notes/images/pi-pre.jpeg)

Assembled:
![image](../../notes/images/pi.jpeg)


### ssh
```console
$ ssh danbev@danbev-pi.local
Linux danbev-pi 6.12.75+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.12.75-1+rpt1 (2026-03-11) aarch64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
```

### Install libraries
```console
$ sudo apt install -y dkms hailo-h10-all
$ sudo reboot
```

### Verify setup
```console
danbev@danbev-pi:~ $ hailortcli fw-control identify
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 5.1.1 (release,app)
Logger Version: 0
Device Architecture: HAILO10H
```

_wip_

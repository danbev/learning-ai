## Scalable Vector Extensions (SVE)


### QEMU installation (Quick Emulator)
On Ubuntu, this does not work on macos, you can install QEMU using the following command:
TODO:
```console
$ sudo apt-get install qemu-user
```

And install a cross compiler:
```console
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

```console
$ qemu-aarch64 --cpu help
Available CPUs:
  a64fx
  cortex-a35
  cortex-a53
  cortex-a55
  cortex-a57
  cortex-a710
  cortex-a72
  cortex-a76
  max
  neoverse-n1
  neoverse-n2
  neoverse-v1
```

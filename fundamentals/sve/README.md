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

### SVE vec_scale example
[vec_scale.cpp](src/vec_scale.cpp) is a standalone reproducer of an issue in
llama.cpp. 

Running the reproducer that shows the issue (need to comment "in" the code for
this):
```console
$ make run-vec-scale
aarch64-linux-gnu-g++ -march=armv8-a+sve -g -static -o bin/vec_scale src/vec_scale.cpp
qemu-aarch64 -cpu max,sve256=on bin/vec_scale
=== GGML vec_scale Leftover Bug Demo ===

SVE elements per vector (svcntw): 8
SVE vector length: 32 bytes (256 bits)
Elements per register (epr): 8
Step size (two registers per loop): 16

Testing with n=25, scale=2.0
Expected: np=16, leftovers=9
Before scaling:
y[ 0] =   0.0
y[ 1] =   1.0
y[ 2] =   2.0
y[ 3] =   3.0
y[ 4] =   4.0
y[ 5] =   5.0
y[ 6] =   6.0
y[ 7] =   7.0
y[ 8] =   8.0
y[ 9] =   9.0
y[10] =  10.0
y[11] =  11.0
y[12] =  12.0
y[13] =  13.0
y[14] =  14.0
y[15] =  15.0
y[16] =  16.0
y[17] =  17.0
y[18] =  18.0
y[19] =  19.0
y[20] =  20.0
y[21] =  21.0
y[22] =  22.0
y[23] =  23.0
y[24] =  24.0
Processed 16 elements in main loop
Leftover elements to process: 9
Create predicate with 16, 25
Predicate lanes:
1 1 1 1 1 1 1 1

After scaling:
y[ 0]:   0.0, expected:   0.0
y[ 1]:   2.0, expected:   2.0
y[ 2]:   4.0, expected:   4.0
y[ 3]:   6.0, expected:   6.0
y[ 4]:   8.0, expected:   8.0
y[ 5]:  10.0, expected:  10.0
y[ 6]:  12.0, expected:  12.0
y[ 7]:  14.0, expected:  14.0
y[ 8]:  16.0, expected:  16.0
y[ 9]:  18.0, expected:  18.0
y[10]:  20.0, expected:  20.0
y[11]:  22.0, expected:  22.0
y[12]:  24.0, expected:  24.0
y[13]:  26.0, expected:  26.0
y[14]:  28.0, expected:  28.0
y[15]:  30.0, expected:  30.0
y[16]:  32.0, expected:  32.0
y[17]:  34.0, expected:  34.0
y[18]:  36.0, expected:  36.0
y[19]:  38.0, expected:  38.0
y[20]:  40.0, expected:  40.0
y[21]:  42.0, expected:  42.0
y[22]:  44.0, expected:  44.0
y[23]:  46.0, expected:  46.0
y[24]:  24.0, expected:  48.0
```
Notice that the last element is not scaled correctly.

Running with the correct code, what is actually in the reproducer:
```console
$ make run-vec-scale 
aarch64-linux-gnu-g++ -march=armv8-a+sve -g -static -o bin/vec_scale src/vec_scale.cpp
qemu-aarch64 -cpu max,sve256=on bin/vec_scale
=== GGML vec_scale Leftover Bug Demo ===

SVE elements per vector (svcntw): 8
SVE vector length: 32 bytes (256 bits)
Elements per register (epr): 8
Step size (two registers per loop): 16

Testing with n=25, scale=2.0
Expected: np=16, leftovers=9
Before scaling:
y[ 0] =   0.0
y[ 1] =   1.0
y[ 2] =   2.0
y[ 3] =   3.0
y[ 4] =   4.0
y[ 5] =   5.0
y[ 6] =   6.0
y[ 7] =   7.0
y[ 8] =   8.0
y[ 9] =   9.0
y[10] =  10.0
y[11] =  11.0
y[12] =  12.0
y[13] =  13.0
y[14] =  14.0
y[15] =  15.0
y[16] =  16.0
y[17] =  17.0
y[18] =  18.0
y[19] =  19.0
y[20] =  20.0
y[21] =  21.0
y[22] =  22.0
y[23] =  23.0
y[24] =  24.0
Processed 16 elements in main loop
Leftover elements to process: 9
Predicate lanes:
1 1 1 1 1 1 1 1 
Predicate lanes:
1 0 0 0 0 0 0 0 

After scaling:
y[ 0]:   0.0, expected:   0.0
y[ 1]:   2.0, expected:   2.0
y[ 2]:   4.0, expected:   4.0
y[ 3]:   6.0, expected:   6.0
y[ 4]:   8.0, expected:   8.0
y[ 5]:  10.0, expected:  10.0
y[ 6]:  12.0, expected:  12.0
y[ 7]:  14.0, expected:  14.0
y[ 8]:  16.0, expected:  16.0
y[ 9]:  18.0, expected:  18.0
y[10]:  20.0, expected:  20.0
y[11]:  22.0, expected:  22.0
y[12]:  24.0, expected:  24.0
y[13]:  26.0, expected:  26.0
y[14]:  28.0, expected:  28.0
y[15]:  30.0, expected:  30.0
y[16]:  32.0, expected:  32.0
y[17]:  34.0, expected:  34.0
y[18]:  36.0, expected:  36.0
y[19]:  38.0, expected:  38.0
y[20]:  40.0, expected:  40.0
y[21]:  42.0, expected:  42.0
y[22]:  44.0, expected:  44.0
y[23]:  46.0, expected:  46.0
y[24]:  48.0, expected:  48.0
```

## Rust Torch (tch) example

### Installation
Download the version for your system from
[pytorch.org](https://pytorch.org/get-started/locally/) and extract it to the
current directory. 
```console
$ wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
$ unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
```

Then set the `LIBTORCH`, and `LD_LIBRARY_PATH` environment variables:
```bash
$ export LIBTORCH=$PWD/libtorch
$ export LD_LIBRARY_PATH=$PWD/libtorch/lib:$LD_LIBRARY_PATH
```
### Running
```bash
$ cargo r
    Finished dev [unoptimized + debuginfo] target(s) in 0.04s
     Running `target/debug/tch-rs-example`
  6
  2
  8
  2
 10
[ CPUIntType{5} ]
```

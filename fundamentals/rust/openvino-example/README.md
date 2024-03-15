## OpenVino Rust example
This is an example of using OpenVino with Rust where the main motivation it to
verify that OpenVino can be used with Rust.

### Prerequisites
OpenVino needs to be installed on your system, I build this using the
instructions in [open-vino.md](../../../open-vino.md).

```console
$ source setenv.sh
[setupvars.sh] OpenVINO environment initialized
```

We also need to download the model used for this example:
```console
$ make download-model
```

## LanceDB Rust example

### Pre-requisites
You need to have protobuf-devel installed or you will get the following error:

```console
$ cargo b
   Compiling zstd-sys v2.0.8+zstd.1.5.5
   Compiling zstd-safe v6.0.6
   Compiling lz4-sys v1.9.4
   Compiling aws-runtime v0.56.1
   Compiling lance v0.7.3
error: failed to run custom build command for `lance v0.7.3`

Caused by:
  process didn't exit successfully: `/home/danielbevenius/work/ai/learning-ai/vector-databases/lancedb/rust/lancedb-example/target/debug/build/lance-829cb93651fa3954/build-script-build` (exit status: 1)
  --- stdout
  cargo:rerun-if-changed=protos

  --- stderr
  Error: Custom { kind: Other, error: "protoc failed: google/protobuf/timestamp.proto: File not found.\nformat.proto:23:1: Import \"google/protobuf/timestamp.proto\" was not found or had errors.\nformat.proto:71:3: \"google.protobuf.Timestamp\" is not defined.\n" }
warning: build failed, waiting for other jobs to finish...
```

protobuf-devel can be installed using the following command:
```console
$ sudo dnf install -y protobuf-devel 
```

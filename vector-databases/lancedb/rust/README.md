## LanceDB Rust example
The goal of this example is to have an equivalent of the Python example in Rust.

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

### Running
```console
$ cargo r -q
LanceDB Rust Example
Opening table "my_table"

         vector     item     price     _distance    
0        [3.0  5.0]  bar     20.0      18434.0
1        [5.0  0.0]  foo     10.0      19801.0
```

## Troubleshooting

### arrow dependency issue
I ran into an issue where I got the following error message:
```console
   Compiling lancedb-example v0.1.0 (/home/danielbevenius/work/ai/learning-ai/vector-databases/lancedb/rust/lancedb-example)
error[E0277]: the trait bound `impl RecordBatchReader + Send + Sync + 'static: arrow_array::record_batch::RecordBatchReader` is not satisfied
   --> src/main.rs:20:45
    |
20  |         let t = db.create_table(table_name, batches, None).await?;
    |                    ------------             ^^^^^^^ the trait `arrow_array::record_batch::RecordBatchReader` is not implemented for `impl RecordBatchReader + Send + Sync + 'static`
    |                    |
    |                    required by a bound introduced by this call
    |
    = help: the following other types implement trait `arrow_array::record_batch::RecordBatchReader`:
              arrow_array::record_batch::RecordBatchIterator<I>
              arrow_csv::reader::BufReader<R>
              arrow_ipc::reader::FileReader<R>
              arrow_ipc::reader::StreamReader<R>
              arrow_json::reader::Reader<R>
              parquet::arrow::arrow_reader::ParquetRecordBatchReader
note: required by a bound in `Database::create_table`
   --> /home/danielbevenius/.cargo/git/checkouts/lancedb-46cbf9f2a5b4f61c/dbf37a0/rust/vectordb/src/database.rs:165:23
    |
162 |     pub async fn create_table(
    |                  ------------ required by a bound in this associated function
...
165 |         batches: impl RecordBatchReader + Send + 'static,
    |                       ^^^^^^^^^^^^^^^^^ required by this bound in `Database::create_table`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `lancedb-example` (bin "lancedb-example") due to previous error
```
After a lot of troubleshooting I was able to track this down to the array
dependency versions. Using version "43.0.0" of `arrow-array` and `arrow-schema`
allowed me to get around this issue. This is the version that is used by
lancedb as well.

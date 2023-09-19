use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};
use std::fs;
use std::sync::Arc;
use vectordb::Database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LanceDB Rust Example");
    let uri = fs::canonicalize("./data/sample-lancedb/")?;
    let uri = uri.to_str().ok_or("Invalid URI")?;
    let db = Database::connect(uri).await?;
    println!("Connected to {:?}", &uri);

    let table_names = db.table_names().await?;
    println!("table_names: {:?}", table_names);
    let table_name = "my_table";
    let table = if table_names.is_empty() {
        println!("Creating table {:?}", table_name);
        let batches = make_test_batches();
        db.create_table(table_name, batches, None).await?
    } else {
        println!("Opening table {:?}", table_name);
        db.open_table(table_name).await?
    };
    println!("table: {:?}", table);
    //println!("db: {:?}", db);
    //db.create_table(table_name).await;

    Ok(())
}

fn make_test_batches() -> impl RecordBatchReader + Send + Sync + 'static {
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
    RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10))],
        )],
        schema,
    )
}

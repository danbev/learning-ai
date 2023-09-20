use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use std::fs;
use std::sync::Arc;
use vectordb::Database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LanceDB Rust Example");
    let uri = fs::canonicalize("./data/sample-lancedb/")?;
    let uri = uri.to_str().ok_or("Invalid URI")?;
    let db = Database::connect(uri).await?;

    let table_names = db.table_names().await?;
    let table_name = "my_table";
    let table = if table_names.is_empty() {
        println!("Creating table {:?}", table_name);
        let schema = Arc::new(Schema::new(vec![
            Field::new("item", DataType::Utf8, false),
            Field::new("price", DataType::Float32, false),
        ]));

        let batches = RecordBatchIterator::new(
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["foo", "bar"])),
                    Arc::new(Float32Array::from(vec![10.0, 20.0])),
                ],
            )],
            schema,
        );
        db.create_table(table_name, batches, None).await?
    } else {
        println!("Opening table {:?}", table_name);
        db.open_table(table_name).await?
    };
    println!("table: {:?}", table);

    let vector = Float32Array::from_iter_values([10.0, 10.0]);
    let mut result = table.search(vector).limit(2).execute().await?;
    let next = result.next().await.unwrap().unwrap();
    println!("next: {:?}", next);
    Ok(())
}

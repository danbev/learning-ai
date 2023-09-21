use arrow_array::types::Float32Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use std::fs;
use std::sync::Arc;
use vectordb::Database;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LanceDB Rust Example");

    let schema = Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        ),
        Field::new("item", DataType::Utf8, false),
        Field::new("price", DataType::Float32, false),
    ]);

    let data = vec![
        Some(vec![Some(0.0), Some(1.0)]),
        Some(vec![Some(3.0), Some(5.0)]),
    ];
    let id_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(data, 2);

    let record_batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_array),
            Arc::new(StringArray::from(vec!["foo", "bar"])),
            Arc::new(Float32Array::from(vec![10.0, 20.0])),
        ],
    )
    .unwrap();
    let batches: Vec<RecordBatch> = vec![record_batch.clone()];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), record_batch.schema());

    let uri = fs::canonicalize("./data/sample-lancedb/")?;
    let uri = uri.to_str().ok_or("Invalid URI")?;
    let db = Database::connect(uri).await?;

    let table_names = db.table_names().await?;
    let table_name = "my_table";
    let table = if table_names.is_empty() {
        println!("Creating table {:?}\n", table_name);
        db.create_table(table_name, reader, None).await?
    } else {
        println!("Opening table {:?}\n", table_name);
        db.open_table(table_name).await?
    };

    let vector = Float32Array::from_iter_values([100.0, 100.0]);
    let mut result = table.search(vector).limit(2).execute().await?;
    let batch_record = result.next().await.unwrap().unwrap();
    let batch_schema = batch_record.schema();
    let mut vector_values = Vec::new();
    let mut item_values = Vec::new();
    let mut price_values = Vec::new();
    let mut distance_values = Vec::new();

    let fields = batch_schema.fields();
    for field in fields {
        let array = batch_record.column_by_name(field.name()).unwrap();
        match array.data_type() {
            DataType::FixedSizeList(_, _) => {
                let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                let values = array.values();
                let values = values.as_any().downcast_ref::<Float32Array>().unwrap();
                vector_values.push(values.slice(0, 2));
                vector_values.push(values.slice(1, 2));
            }
            DataType::Float32 => {
                let array = array.as_any().downcast_ref::<Float32Array>().unwrap();
                for v in array.iter() {
                    if field.name() == "price" {
                        price_values.push(v.unwrap());
                    } else {
                        distance_values.push(v.unwrap());
                    }
                }
            }
            DataType::Utf8 => {
                let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                for v in array.iter() {
                    item_values.push(v.unwrap());
                }
            }
            _ => {}
        }
    }
    print!("        ");
    for field in fields {
        print!(" {}    ", field.name());
    }
    println!("");
    let zipped = vector_values
        .iter()
        .zip(item_values.iter())
        .zip(price_values.iter().zip(distance_values.iter()));
    for (i, ((vector, item), (price, distance))) in zipped.enumerate() {
        print!(
            "{}        [{:.1}  {:.1}]  {}     {:.1}      {:.1}\n",
            i,
            vector.values().get(0).unwrap(),
            vector.values().get(1).unwrap(),
            item,
            price,
            distance
        );
    }
    Ok(())
}

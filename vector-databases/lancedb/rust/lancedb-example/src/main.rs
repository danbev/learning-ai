use arrow_array::Float32Array;
use std::fs;
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
    let _table_name = "my_table";
    let table = if table_names.is_empty() {
        let vector = Float32Array::from_iter_values([3.1, 4.1]);
        println!("vector: {:?}", vector);
        //let t = db.create_table(table_name, , None).await?;
        "create"
    } else {
        "open"
    };
    println!("table: {:?}", table);
    //println!("db: {:?}", db);
    //db.create_table(table_name).await;

    Ok(())
}

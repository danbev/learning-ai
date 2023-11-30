pub mod data;
use std::io::{self};

fn main() -> io::Result<()> {
    let data = data::Data::new("names.txt");
    println!("chars {:?}", data.chars());
    Ok(())
}

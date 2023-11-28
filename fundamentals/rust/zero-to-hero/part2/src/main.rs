use std::io::{self};

use micrograd::data::Data;

fn main() -> io::Result<()> {
    let data = Data::new("names.txt");
    println!("chars {:?}", data.chars());
    Ok(())
}

mod data;
use data::{Data, WORD_BOUNDARY as dot};
use std::io::{self};
use tch::{kind::Kind, Device, IndexOp, Tensor};

fn main() -> io::Result<()> {
    let data = Data::new("names.txt")?;
    println!("chars: {:?}", data.chars());

    let counts = Tensor::zeros(&[27, 27], (Kind::Int, Device::Cpu));

    // Fill the tensor with bigram counts from the data
    for w in data.words() {
        let chs = format!("{dot}{w}{dot}");
        for (ch1, ch2) in chs.chars().zip(chs[1..].chars()) {
            let ix1 = data.stoi(ch1);
            let ix2 = data.stoi(ch2);
            let mut count = counts.i((ix1 as i64, ix2 as i64));
            count += 1;
        }
    }

    for i in 0..27 {
        for j in 0..27 {
            let chstr = format!("{}{}", data.itos(i), data.itos(j));
            let v = counts.i((i as i64, j as i64)).int64_value(&[]);
            print!("{chstr}={v}, ");
        }
        println!("");
    }

    Ok(())
}

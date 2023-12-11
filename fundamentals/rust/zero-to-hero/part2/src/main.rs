mod data;
use data::{Data, WORD_BOUNDARY as dot};
use ndarray::Array1;
use plotters::prelude::*;
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

    create_plot(&counts, &data, "part2-count-grid");

    // Set the seed for reproducibility of values in the next part.
    tch::manual_seed(2147483647);

    Ok(())
}

fn create_plot(
    tensor: &Tensor,
    data: &Data,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = (tensor.size()[0] as usize, tensor.size()[1] as usize);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut values = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            xs.push(format!("{}{}", data.itos(i), data.itos(j)));
            let v = tensor.i((i as i64, j as i64)).int64_value(&[]);
            ys.push(v);
            values.push(tensor.double_value(&[i as i64, j as i64]));
        }
    }

    let xs_array = Array1::from(xs);
    let ys_array = Array1::from(ys);

    let file = format!("plots/{}.png", file_name);
    let root_area = BitMapBackend::new(&file, (800, 800)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let font_size = 10;
    let text_offset = 5;

    for (idx, (label, value)) in xs_array.iter().zip(ys_array.iter()).enumerate() {
        let row = idx / cols;
        let col = idx % cols;
        let value = value.to_string();

        let (x_range, y_range) = root_area.get_pixel_range();
        let cell_width = (x_range.end - x_range.start) as u32 / cols as u32;
        let cell_height = (y_range.end - y_range.start) as u32 / rows as u32;

        let text_style = TextStyle::from(("sans-serif", font_size).into_font());
        let cell = root_area.titled(&value, &text_style)?.shrink(
            (col as u32 * cell_width, row as u32 * cell_height),
            (cell_width, cell_height),
        );
        cell.fill(&WHITE)?;

        let label_size = cell.estimate_text_size(label, &text_style)?;
        let value_size = cell.estimate_text_size(&value, &text_style)?;

        let label_x_pos = (cell_width / 2) as u32 - (label_size.0 / 2) as u32;
        let label_y_pos = (cell_height / 2) as u32 - label_size.1 as u32 / 2;
        let value_x_pos = label_x_pos;
        let value_y_pos = label_y_pos + label_size.1 as u32 + 3; // Adding a small offset for value text

        let label_pos = (label_x_pos as i32, label_y_pos as i32);
        let value_pos = (value_x_pos as i32, value_y_pos as i32);

        cell.draw(&Text::new(label.to_string(), label_pos, &text_style))?;
        cell.draw(&Text::new(value.clone(), value_pos, &text_style))?;

        /*
        println!(
            "idx: {}, row: {}, col: {}, cell_width: {}, cell_height: {}, label_pos: {:?}, value_pos: {:?}, label: {:?}, value: {:?}",
            idx, row, col, cell_width, cell_height, value_pos, label_pos, label, &value
        );
        */
    }

    root_area.present()?;

    Ok(())
}

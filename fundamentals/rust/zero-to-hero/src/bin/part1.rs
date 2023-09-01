use ndarray::prelude::*;
use ndarray::Array;
use plotpy::{Curve, Plot};
use std::io::{self};

fn f(xs: Array1<f64>) -> Array1<f64> {
    xs.mapv(|x| 3.0 * (x * x) - 4.0 * x + 5.0)
}

fn main() -> io::Result<()> {
    // -----------------  intro ---------------------------

    // Tryout the function f:
    let x = f(array![3.0]);
    println!("x = {x}");

    // Generate example input data:
    let xs = Array::range(-5., 5., 0.25);
    println!("xs = {xs:?}");

    // Show that we can call the function with the input data:
    let ys = f(xs.clone());
    println!("ys = {ys:?}");

    plot(&xs, &ys, "part1_intro");

    // -----------------  micrograd overview ---------------------------
    //TODO: add micrograd overview here

    Ok(())
}

fn plot(xs: &Array1<f64>, ys: &Array1<f64>, name: &str) {
    let mut curve = Curve::new();

    // draw curve
    //let x = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    //let y = &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 3.5, 3.5, 3.5];
    curve.draw(&xs.to_vec(), &ys.to_vec());

    // configure plot
    let mut plot = Plot::new();
    plot.set_subplot(2, 2, 1)
        .set_horizontal_gap(0.1)
        .set_vertical_gap(0.2)
        .set_gaps(0.3, 0.4)
        .set_equal_axes(true)
        .set_hide_axes(false)
        .set_range(-1.0, 1.0, -1.0, 1.0)
        .set_range_from_vec(&[0.0, 1.0, 0.0, 1.0])
        .set_xmin(0.0)
        .set_xmax(1.0)
        .set_ymin(0.0)
        .set_ymax(1.0)
        .set_xrange(0.0, 1.0)
        .set_yrange(0.0, 1.0)
        .set_num_ticks_x(0)
        .set_num_ticks_x(8)
        .set_num_ticks_y(0)
        .set_num_ticks_y(5)
        .set_label_x("x-label")
        .set_label_y("y-label")
        .set_labels("x", "y")
        .clear_current_axes();
    plot.clear_current_figure();
    plot.set_title("Plot of f(x)")
        .set_frame_borders(false)
        .set_frame_borders(true)
        .set_frame_borders(false);
    plot.grid_and_labels("x", "y");
    plot.add(&curve);
    let _ = plot.save(&format!("./plots/{name}.svg")).unwrap();
}

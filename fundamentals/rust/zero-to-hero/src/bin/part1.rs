use ndarray::prelude::*;
use ndarray::Array;
use std::io::{self};

fn f(xs: Array1<f64>) -> Array1<f64> {
    xs.mapv(|x| 3.0 * (x * x) - 4.0 * x + 5.0)
}

fn main() -> io::Result<()> {
    // Tryout the function f:
    let x = f(array![3.0]);
    println!("x = {x}");

    // Generate example input data:
    let xs = Array::range(-5., 5., 0.25);
    println!("xs = {xs:?}");

    // Show that we can call the function with the input data:
    let ys = f(xs);
    println!("ys = {ys:?}");

    Ok(())
}

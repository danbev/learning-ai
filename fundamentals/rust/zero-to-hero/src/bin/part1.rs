use ndarray::prelude::*;
use ndarray::Array;
use plotpy::{Curve, Plot};
use std::io::{self};

fn f(xs: Array1<f64>) -> Array1<f64> {
    xs.mapv(|x| 3.0 * x * x - 4.0 * x + 5.0)
}

/// The derivative of f() with respect to x.
#[allow(dead_code)]
fn f_prime(xs: Array1<f64>) -> Array1<f64> {
    xs.mapv(|x| 6.0 * x - 4.0)
}

fn main() -> io::Result<()> {
    // -----------------  intro ---------------------------
    println!("f(x) = 3.0 * x * x - 4.0 * x + 5.0");

    println!("\nLets tryout the function f:");
    println!("f(3.0) = {}", f(array![3.0])[0]);

    println!("\nLets generate some input data:");
    let xs = Array::range(-5., 5., 0.25);
    println!("xs = {xs:?}");

    println!("\nLets try invoking f(xs):");
    let ys = f(xs.clone());
    println!("ys = {ys:?}");

    plot(&xs, &ys, "part1_intro");

    // We can decrease this value, the nudge to be closer and closer to zero.
    println!("\nLets take a look at when the derivative:");
    let h = 0.00000001;
    let x = 3.0;
    println!("h = {h}");
    println!("x = {}", x);
    println!("f(x + h) =  {}", f(array![x + h])[0]);
    println!(
        "f(x + h) - f(x) / h = {}",
        (f(array![x + h])[0] - f(array![x])[0]) / h
    );
    // These values won't be exactly equal but the smaller h becomes the closer
    // they will be.
    println!("f_prime(x) =  {}", f_prime(array![x])[0]);

    println!("\nLets take a look at when the derivative is negative:");
    let x = -3.0;
    println!("x = {}", x);
    println!(
        "f(x + h) - f(x) / h = {}",
        (f(array![x + h])[0] - f(array![x])[0]) / h
    );

    // Show when the deriative is zero:
    println!("\nLets take a look at when the derivative is zero:");
    let x = 2.0 / 3.0;
    println!("x = {} (2/3)", x);
    println!(
        "f(x + h) - f(x) / h = {}",
        (f(array![x + h])[0] - f(array![x])[0]) / h
    );

    println!("\nNow lets take a look at a more complex example:");
    let a = 2.0;
    let b = -3.0;
    let c = 10.0;
    let d = a * b + c;
    println!("a = {a:.1}");
    println!("b = {b:.1}");
    println!("c = {c:.1}");
    println!("d = {d:.1}");

    let h = 0.0001;
    let mut a = 2.0;
    let b = -3.0;
    let c = 10.0;

    // d1 is our original function that we will use as a example
    let d1 = a * b + c;
    a += h;
    // d2 is the function with a nudged/dumped a little.
    let d2 = a * b + c;

    println!("\nDeriviative with respect to a:");
    println!("d1 (original function) = {d1:.6}");
    println!("d2 (nudged a         ) = {d2:.6}");
    println!("slope (d2 - d1) / h = {}", (d2 - d1) / h);

    let a = 2.0;
    let mut b = -3.0;
    let c = 10.0;
    let d1 = a * b + c;
    b += h;
    let d2 = a * b + c;
    println!("\nDeriviative with respect to b:");
    println!("d1 (original function) = {d1:.6}");
    println!("d2 (nudged b         ) = {d2:.6}");
    println!("slope (d2 - d1) / h = {}", (d2 - d1) / h);

    let a = 2.0;
    let b = -3.0;
    let mut c = 10.0;
    let d1 = a * b + c;
    c += h;
    let d2 = a * b + c;
    println!("\nDeriviative with respect to b:");
    println!("d1 (original function) = {d1:.6}");
    println!("d2 (nudged c         ) = {d2:.6}");
    println!("slope (d2 - d1) / h = {}", (d2 - d1) / h);

    // -----------------  micrograd overview ---------------------------
    //TODO: add micrograd overview here

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct Value {
        data: f64,
        children: Vec<Value>,
        operation: Option<String>,
    }
    // Some of the comments below have been kept as they were prompts for
    // copilot to generated the code.

    // Add a new constructor for Value which takes a single f64.
    impl Value {
        fn new(data: f64) -> Self {
            Value {
                data,
                children: Vec::new(),
                operation: None,
            }
        }
    }

    // Add a new_with_children constructor for Value which takes a single f64, and a
    // parameter named 'children' of type Vec and that contains Values
    // as the element types.
    impl Value {
        fn new_with_children(data: f64, children: Vec<Value>, op: String) -> Self {
            Value {
                data,
                children,
                operation: Some(op),
            }
        }
    }

    // Add Add trait implementation for Value and add use statement
    use std::ops::Add;
    impl Add for Value {
        type Output = Value;
        fn add(self, other: Value) -> Self::Output {
            Value::new_with_children(self.data + other.data, vec![self, other], "+".to_string())
        }
    }

    // Add Sub trait implementation for Value and add use statement
    use std::ops::Sub;
    impl Sub for Value {
        type Output = Value;
        fn sub(self, other: Value) -> Self::Output {
            Value::new_with_children(self.data - other.data, vec![self, other], "-".to_string())
        }
    }

    // Add Mul trait implementation for Value and add use statement
    use std::ops::Mul;
    impl Mul for Value {
        type Output = Value;
        fn mul(self, other: Value) -> Self::Output {
            Value::new_with_children(self.data * other.data, vec![self, other], "*".to_string())
        }
    }

    // Implement the Display trait for Value in the format Value(data) and
    // include any necessary use statements
    use std::fmt;
    impl fmt::Display for Value {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Value({})", self.data)
        }
    }

    // Implement get functions for children and operation
    impl Value {
        fn children(&self) -> &Vec<Value> {
            &self.children
        }
        fn operation(&self) -> &Option<String> {
            &self.operation
        }
    }

    let a = Value::new(2.0);
    println!("a = {}", a);
    let b = Value::new(-3.0);
    println!("{a} + {b} = {}", a.clone() + b.clone());
    println!("{a} - {b} = {}", a.clone().sub(b.clone()));
    println!("{a} * {b} = {}", a.clone() * b.clone());
    let c = Value::new(10.0);
    println!("{a} * {b} + {c} = {}", a.clone() * b.clone() + c.clone());

    Ok(())
}

fn plot(xs: &Array1<f64>, ys: &Array1<f64>, name: &str) {
    let mut curve = Curve::new();

    curve.draw(&xs.to_vec(), &ys.to_vec());

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

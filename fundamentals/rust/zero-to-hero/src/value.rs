use std::cell::RefCell;
use uuid::Uuid;

#[allow(dead_code)]
#[derive(Debug)]
struct Value<'a> {
    id: Uuid,
    data: RefCell<f64>,
    label: Option<String>,
    children: Vec<&'a Value<'a>>,
    operation: Option<Operation>,
    grad: RefCell<f64>,
}

impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

// Implement Eq for Value
impl<'a> Eq for Value<'a> {}

use std::hash::{Hash, Hasher};
impl<'a> Hash for Value<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Tanh,
    Exp,
    Pow,
}

// Implement as_str function for Operation enum
#[allow(dead_code)]
impl Operation {
    fn as_str(&self) -> &'static str {
        match self {
            Operation::Add => "+",
            Operation::Sub => "-",
            Operation::Mul => "*",
            Operation::Div => "/",
            Operation::Tanh => "tanh",
            Operation::Exp => "exp",
            Operation::Pow => "pow",
        }
    }
}

// Some of the comments below have been kept as they were prompts for
// copilot to generated the code.

// Add a new constructor for Value which takes a single f64.
#[allow(dead_code)]
impl<'a> Value<'a> {
    fn new(data: f64) -> Self {
        Value {
            id: Uuid::new_v4(),
            data: RefCell::new(data),
            label: None,
            children: Vec::new(),
            operation: None,
            grad: RefCell::new(0.0), // we initialize the gradient to 0.0
        }
    }
    fn new_with_label(data: f64, label: &str) -> Self {
        Value {
            id: Uuid::new_v4(),
            data: RefCell::new(data),
            label: Some(label.to_string()),
            children: Vec::new(),
            operation: None,
            grad: RefCell::new(0.0), // we initialize the gradient to 0.0
        }
    }
}

impl<'a> Value<'a> {
    fn backward(&self) {
        match self.operation {
            Some(Operation::Add) => {
                // Think of this as d = c + e
                // Then &self is d and self.children is (c, e), and since
                // addition passes through the gradient we can just set the
                // gradients of the children to the gradient of d.
                let lhs = self.children[0];
                let rhs = self.children[1];
                // If we have have a + a then both lhs and rhs will be then
                // same value so we accumulate the gradient.
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Some(Operation::Sub) => {
                let lhs = self.children[0];
                let rhs = self.children[1];
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Some(Operation::Mul) => {
                let lhs = self.children[0];
                let rhs = self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Div) => {
                let lhs = self.children[0];
                let rhs = self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Tanh) => {
                let lhs = self.children[0];
                *lhs.grad.borrow_mut() += 1.0 - self.data.borrow().powf(2.0) * *self.grad.borrow();
            }
            Some(Operation::Exp) => {
                let lhs = self.children[0];
                // e^x * dx/dx = e^x
                *lhs.grad.borrow_mut() += *self.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Pow) => {
                // In an attempt to visualize this, say we have the
                // following:
                // let a = Value::new(2.0);
                // let b = Value::new(4.0);
                // let c = &a.pow(&b);
                //  +-------------------+       +--------------------+
                //  |a, data: 2, grad:  |-------|c, data: 16, grad:  |
                //  +-------------------+       +--------------------+
                //  +-------------------+      /
                //  |b, data: 4, grad:  |-----+
                //  +-------------------+
                // Now if we call c.backward() we want to compute the
                // derivitives of a and b with respect to c.
                // c will be self, (so the derivative with respect to itself is 1.0)
                // a will be lhs (base)
                // b will be rhs (exponent)
                let lhs = self.children[0];
                let rhs = self.children[1];
                let base = *lhs.data.borrow();
                let exponent = *rhs.data.borrow();
                println!("self: {}", *self.data.borrow());
                println!("base: {}", base);
                println!("exponent: {}", exponent);
                // Here we use the power rule:
                *lhs.grad.borrow_mut() +=
                    exponent * (base.powf(exponent - 1.0)) * *self.grad.borrow();
            }
            None => {
                //println!("No backward for you! {}", self.label.as_ref().unwrap());
            }
        }
    }
}

use std::collections::VecDeque;
#[allow(dead_code)]
impl<'a> Value<'a> {
    fn topological_sort(
        root: &'a Value<'a>,
        visited: &mut HashSet<&'a Value<'a>>,
        stack: &mut VecDeque<&'a Value<'a>>,
    ) {
        visited.insert(root);

        for child in root.children.iter() {
            if !visited.contains(child) {
                Self::topological_sort(child, visited, stack);
            }
        }

        stack.push_front(root);
    }

    fn topological_order(value: &'a Value<'a>) -> VecDeque<&'a Value<'a>> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();
        Self::topological_sort(&value, &mut visited, &mut stack);
        stack
    }

    fn backwards(value: &'a Value<'a>) {
        *value.grad.borrow_mut() = 1.0;
        // Now lets do the backpropagation using the topological order.
        let order = Value::topological_order(&value);
        println!("topological order:");
        for (i, node) in order.iter().enumerate() {
            println!("{}: {:?}", i, node.label);
            node.backward();
        }
    }
}

// Add a new_with_children constructor for Value which takes a single f64, and a
// parameter named 'children' of type Vec and that contains Values
// as the element types.
impl<'a> Value<'a> {
    fn new_with_children(
        data: f64,
        label: Option<String>,
        lhs: &'a Value<'a>,
        rhs: Option<&'a Value<'a>>,
        op: Operation,
    ) -> Self {
        let children = match rhs {
            Some(rhs) => vec![lhs, rhs],
            None => vec![lhs],
        };
        Value {
            id: Uuid::new_v4(),
            data: RefCell::new(data),
            label,
            children,
            operation: Some(op),
            grad: RefCell::new(0.0),
        }
    }
}

// Add Add trait implementation for Value and add use statement
use std::ops::Add;
impl<'a> Add<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn add(self, other: &'a Value<'a>) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() + *other.data.borrow(),
            None,
            self,
            Some(other),
            Operation::Add,
        )
    }
}

impl<'a> Add<&'a Value<'a>> for f64 {
    type Output = Value<'a>;

    fn add(self, other: &'a Value<'a>) -> Self::Output {
        Value::new_with_children(
            self + *other.data.borrow(),
            None,
            other,
            None,
            Operation::Add,
        )
    }
}

impl<'a> Add<f64> for &'a Value<'a> {
    type Output = Value<'a>;

    fn add(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() + other,
            None,
            self,
            None,
            Operation::Add,
        )
    }
}

// Add Sub trait implementation for Value and add use statement
use std::ops::Sub;
impl<'a, 'b: 'a> Sub<&'b Value<'b>> for &'a Value<'a> {
    type Output = Value<'a>;
    fn sub(self, other: &'b Value<'b>) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() - *other.data.borrow(),
            None,
            self,
            Some(other),
            Operation::Sub,
        )
    }
}

impl<'a> Sub<&'a Value<'a>> for f64 {
    type Output = Value<'a>;

    fn sub(self, other: &'a Value<'a>) -> Self::Output {
        Value::new_with_children(
            self - *other.data.borrow(),
            None,
            other,
            None,
            Operation::Sub,
        )
    }
}

impl<'a> Sub<f64> for &'a Value<'a> {
    type Output = Value<'a>;

    fn sub(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() - other,
            None,
            self,
            None,
            Operation::Sub,
        )
    }
}

// Add Mul trait implementation for Value and add use statement
use std::ops::Mul;
impl<'a, 'b: 'a> Mul<&'b Value<'b>> for &'a Value<'a> {
    type Output = Value<'a>;
    fn mul(self, other: &'b Value<'b>) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() * *other.data.borrow(),
            None,
            self,
            Some(other),
            Operation::Mul,
        )
    }
}

impl<'a> Mul<&'a Value<'a>> for f64 {
    type Output = Value<'a>;

    fn mul(self, other: &'a Value<'a>) -> Self::Output {
        Value::new(self * *other.data.borrow())
    }
}

impl<'a> Mul<f64> for &'a Value<'a> {
    type Output = Value<'a>;

    fn mul(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() * other,
            None,
            self,
            None,
            Operation::Mul,
        )
    }
}

use std::ops::Div;
impl<'a> Div<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn div(self, other: &'a Value<'a>) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() / *other.data.borrow(),
            None,
            self,
            Some(other),
            Operation::Div,
        )
    }
}

impl<'a> Div<&'a Value<'a>> for f64 {
    type Output = Value<'a>;

    fn div(self, other: &'a Value<'a>) -> Self::Output {
        Value::new_with_children(
            self / *other.data.borrow(),
            None,
            other,
            None,
            Operation::Div,
        )
    }
}

impl<'a> Div<f64> for &'a Value<'a> {
    type Output = Value<'a>;

    fn div(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() / other,
            None,
            self,
            None,
            Operation::Div,
        )
    }
}

// Implement the Display trait for Value in the format Value(data) and
// include any necessary use statements
use std::fmt;
impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Value(data={}, label: {}",
            *self.data.borrow(),
            self.label.as_ref().unwrap_or(&"".to_string())
        )?;
        if &self.children.len() > &0 {
            write!(f, ", lhs={}", *self.children[0].data.borrow())?;
            if &self.children.len() == &2 {
                write!(f, ", rhs={}", *self.children[1].data.borrow())?;
            }
            write!(f, ", op=\"{:?}\"", &self.operation.as_ref().unwrap())?;
        }
        write!(f, ", grad={})", self.grad.borrow())
    }
}

use std::collections::HashSet;
use std::f64;
#[allow(dead_code)]
impl<'a> Value<'a> {
    fn operation(&self) -> Option<Operation> {
        self.operation.clone()
    }

    fn label(&mut self, label: &str) {
        self.label = Some(label.to_string())
    }

    fn tanh(&self) -> Value {
        let x = *self.data.borrow();
        //
        // sinh(x) = (e^x - e^-x) / 2
        //
        // cosh(x) = (e^x + e^-x) / 2
        //
        //         sinh(x)    e^x - e^-x
        // tanh =  ------- = -----------
        //         cosh(x)    e^x + e^-x
        let t = (f64::exp(x) - f64::exp(-x)) / (f64::exp(x) + f64::exp(-x));
        println!("tanh({}) = {}", x, t);
        let t = (f64::exp(2.0 * x) - 1.0) / (f64::exp(2.0 * x) + 1.0);
        println!("tanh({}) = {}", x, t);
        Value::new_with_children(t, None, self, None, Operation::Tanh)
    }

    fn exp(&self) -> Value {
        let x = *self.data.borrow();
        let e = f64::exp(x);
        println!("exp({}) = {}", x, e);
        Value::new_with_children(e, None, self, None, Operation::Exp)
    }

    fn pow(&self, x: &'a Value<'a>) -> Value {
        println!("pow({}, {})", *self.data.borrow(), x);
        Value::new_with_children(
            f64::powf(*self.data.borrow(), *x.data.borrow()),
            None,
            self,
            Some(x),
            Operation::Pow,
        )
    }
}

#[allow(dead_code)]
impl Value<'_> {
    fn dot(&self) -> String {
        let mut out = "digraph {\n".to_string();
        out += "graph [rankdir=LR]\n";
        let mut stack = vec![self];
        let mut seen = HashSet::new();

        while let Some(node) = stack.pop() {
            let node_ptr = node as *const _;
            if seen.contains(&node_ptr) {
                continue;
            }

            let node_id = node_ptr as usize;

            let label_str = |node: &Value| -> String {
                match &node.label {
                    Some(l) => format!("{l}:"),
                    None => "".to_string(),
                }
            };
            out += &format!(
                "  \"{}\" [label=\"{} value: {:.4}, grad: {:.4}\" shape=record]\n",
                node_ptr as usize,
                label_str(node),
                *node.data.borrow(),
                node.grad.borrow(),
            );

            seen.insert(node_ptr);

            if !&node.children.is_empty() {
                let op_id = format!("{}{}", node_id, node.operation.as_ref().unwrap().as_str());
                let lhs_id = node.children[0] as *const _ as usize;

                out += &format!(
                    "  \"{}\" [label=\"{}\"]\n",
                    op_id,
                    node.operation.as_ref().unwrap().as_str().to_string()
                );
                out += &format!("  \"{}\" -> \"{}\"\n", op_id, node_id,);

                out += &format!("  \"{}\" -> \"{}\"\n", lhs_id, op_id,);
                if &node.children.len() == &2 {
                    let rhs_id = node.children[1] as *const _ as usize;
                    out += &format!("  \"{}\" -> \"{}\"\n", rhs_id, op_id);
                    stack.push(&node.children[1]);
                };

                stack.push(&*node.children[0]);
            }
        }

        out += "}\n";
        out
    }
}

#[cfg(test)]
use approx::assert_abs_diff_eq;

#[test]
fn test_add() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = &a + &b;

    assert_eq!(*c.data.borrow(), 3.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_rhs_float() {
    let a = Value::new(1.0);
    let c = &a + 4.0;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_lhs_float() {
    let a = Value::new(1.0);
    let c = 4.0 + &a;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_backwards() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = &a + &b;
    Value::backwards(&c);

    assert_eq!(*c.grad.borrow(), 1.0);
    assert_eq!(*a.grad.borrow(), 1.0);
    assert_eq!(*b.grad.borrow(), 1.0);
}

// Add a test to test subtraction
#[test]
fn test_sub() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = &a - &b;

    assert_eq!(*c.data.borrow(), -1.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Sub));
}

#[test]
fn test_sub_lhs_float() {
    let a = Value::new(8.0);
    let c = 4.0 - &a;

    assert_eq!(*c.data.borrow(), -4.0);
    assert_eq!(c.operation, Some(Operation::Sub));
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 8.0);
}

#[test]
fn test_sub_rhs_float() {
    let a = Value::new(8.0);
    let c = &a - 4.0;

    assert_eq!(*c.data.borrow(), 4.0);
    assert_eq!(c.operation, Some(Operation::Sub));
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 8.0);
}

// Add a test for testing division
#[test]
fn test_div() {
    let a = Value::new(10.0);
    let b = Value::new(2.0);
    let c = &a / &b;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 10.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Div));
}

#[test]
fn test_div_lhs_float() {
    let a = Value::new(2.0);
    let c = 10.0 / &a;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Div));
}

#[test]
fn test_div_rhs_float() {
    let a = Value::new(10.0);
    let c = &a / 2.0;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 10.0);
    assert_eq!(c.operation, Some(Operation::Div));
}

#[test]
fn test_tanh_backwards() {
    let x1 = Value::new_with_label(2.0, "x1");
    let x2 = Value::new_with_label(0.0, "x2");
    let w1 = Value::new_with_label(-3.0, "w1");
    let w2 = Value::new_with_label(1.0, "w2");
    let b = Value::new_with_label(6.8813735870195432, "b");
    let mut x1w1 = &x1 * &w1;
    x1w1.label("x1*w1");
    let mut x2w2 = &x2 * &w2;
    x2w2.label("x2*w2");
    let mut x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.label("x1w1x + 2w2");
    let mut n = &x1w1x2w2 + &b;
    n.label("n");
    let mut o = n.tanh();
    o.label("o");
    Value::backwards(&o);

    assert_eq!(*o.grad.borrow(), 1.0);
    assert_abs_diff_eq!(*n.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*b.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1w1x2w2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1w1.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x2w2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1.grad.borrow(), -1.5, epsilon = 1e-1);
    assert_abs_diff_eq!(*w1.grad.borrow(), 1.0);
    assert_abs_diff_eq!(*x2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*w2.grad.borrow(), 0.0);
}

#[test]
fn test_tanh_backwards_decomposed() {
    let x1 = Value::new_with_label(2.0, "x1");
    let x2 = Value::new_with_label(0.0, "x2");
    let w1 = Value::new_with_label(-3.0, "w1");
    let w2 = Value::new_with_label(1.0, "w2");
    let b = Value::new_with_label(6.8813735870195432, "b");
    let mut x1w1 = &x1 * &w1;
    x1w1.label("x1*w1");
    let mut x2w2 = &x2 * &w2;
    x2w2.label("x2*w2");
    let mut x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.label("x1w1x + 2w2");
    let mut n = &x1w1x2w2 + &b;
    n.label("n");
    // Here we want to use the following formula for deriving tanh:
    //         e²ˣ - 1
    // tanh =  ----------
    //         e²ˣ + 1
    //
    //let binding = &n * 2.0;
    //let e = binding.exp();
    //println!("e.children[0]: {}", &e.children[0]);
    println!("n: {}", n);
    let e_two_exp = &n * 2.0;
    let e_two_exp = e_two_exp.exp();
    let e_minus_one = &e_two_exp - 1.0;
    let e_plus_one = &e_two_exp + 1.0;
    let mut o = &e_minus_one / &e_plus_one;
    o.label("o");
    /*
    println!("o: {}", &o);
    println!("o.children: {}", &o.children[0]);
    println!("o.children: {}", &o.children[0].children[0]);
    Value::backwards(&o);

    assert_eq!(*o.grad.borrow(), 1.0);
    assert_eq!(*e_plus_one.grad.borrow(), 4.828427);

    assert_abs_diff_eq!(*n.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*b.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1w1x2w2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1w1.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x2w2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*x1.grad.borrow(), -1.5, epsilon = 1e-1);
    assert_abs_diff_eq!(*w1.grad.borrow(), 1.0);
    assert_abs_diff_eq!(*x2.grad.borrow(), 0.5);
    assert_abs_diff_eq!(*w2.grad.borrow(), 0.0);
    */
}

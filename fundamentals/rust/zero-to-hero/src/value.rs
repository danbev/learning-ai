use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use uuid::Uuid;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Value {
    id: Uuid,
    label: Option<String>,
    data: RefCell<f64>,
    children: Vec<Rc<Value>>,
    op: Operation,
    grad: RefCell<f64>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Tanh,
    Exp,
    Pow,
    None,
}

#[allow(dead_code)]
impl Operation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Operation::Add => "+",
            Operation::Sub => "-",
            Operation::Mul => "*",
            Operation::Div => "/",
            Operation::Tanh => "tanh",
            Operation::Exp => "exp",
            Operation::Pow => "pow",
            Operation::None => "N/A",
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value {
            id: Uuid::new_v4(),
            label: None,
            data: RefCell::new(0.0),
            children: vec![],
            op: Operation::None,
            grad: RefCell::new(0.0),
        }
    }
}

#[allow(dead_code)]
impl Value {
    pub fn new(data: f64) -> Self {
        Value {
            data: RefCell::new(data),
            ..Default::default()
        }
    }

    pub fn new_with_label(data: f64, label: &str) -> Self {
        Value {
            label: Some(label.to_string()),
            data: RefCell::new(data),
            ..Default::default()
        }
    }

    pub fn new_with_children(data: f64, children: Vec<Rc<Value>>, op: Operation) -> Self {
        Value {
            data: RefCell::new(data),
            children,
            op,
            ..Default::default()
        }
    }

    pub fn data(&self) -> f64 {
        *self.data.borrow()
    }

    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    pub fn set_label(&mut self, label: &str) {
        self.label = Some(label.to_string());
    }

    pub fn grad(&self) -> f64 {
        *self.grad.borrow()
    }

    pub fn op(&self) -> Operation {
        self.op.clone()
    }

    pub fn topological_sort(
        root: &Rc<Value>,
        visited: &mut HashSet<Uuid>,
        stack: &mut VecDeque<Rc<Value>>,
    ) {
        visited.insert(root.id);
        for child in root.children.iter() {
            if !visited.contains(&child.id) {
                Self::topological_sort(child, visited, stack);
            }
        }
        stack.push_front(Rc::clone(root));
    }

    fn topological_order(root: Rc<Value>) -> VecDeque<Rc<Value>> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();
        Self::topological_sort(&root, &mut visited, &mut stack);
        stack
    }

    pub fn backwards(root: Rc<Value>) {
        *root.grad.borrow_mut() = 1.0;
        // Now lets do the backpropagation using the topological order.
        let order = Self::topological_order(root);
        for (i, node) in order.iter().enumerate() {
            println!("{}:{:?} {:?}", i, node.label, node.data.borrow());
            node.backward();
        }
    }

    fn backward(&self) {
        match self.op {
            Operation::Add => {
                println!("Add for data: {:?}", self.data.borrow());
                // Think of this as c = a + b
                // Then &self is c and self.children are (a, b), and since
                // addition passes through the gradient we can just set the
                // gradients of the children to the gradient of c.
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                // If we have have a + a then both lhs and rhs will be then
                // same value so we accumulate the gradient.
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Operation::Sub => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Operation::Mul => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Operation::Div => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Operation::Tanh => {
                let lhs = &self.children[0];
                *lhs.grad.borrow_mut() += 1.0 - self.data.borrow().powf(2.0) * *self.grad.borrow();
            }
            Operation::Exp => {
                let lhs = &self.children[0];
                // e^x * dx/dx = e^x
                *lhs.grad.borrow_mut() += *self.data.borrow() * *self.grad.borrow();
            }
            Operation::Pow => {
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
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                let base = *lhs.data.borrow();
                let exponent = *rhs.data.borrow();
                println!("self: {}", *self.data.borrow());
                println!("base: {}", base);
                println!("exponent: {}", exponent);
                // Here we use the power rule:
                *lhs.grad.borrow_mut() +=
                    exponent * (base.powf(exponent - 1.0)) * *self.grad.borrow();
            }
            Operation::None => {
                return;
            }
        }
    }

    pub fn add_f64(value: Rc<Self>, other: f64) -> Rc<Self> {
        Rc::new(Self {
            id: Uuid::new_v4(),
            data: RefCell::new(*value.data.borrow() + other),
            children: vec![value.clone(), Rc::new(Self::new(other))],
            op: Operation::Add,
            ..Default::default()
        })
    }

    fn tanh(&self) -> Rc<Self> {
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
        Rc::new(Self {
            id: Uuid::new_v4(),
            data: RefCell::new(t),
            children: vec![Rc::new(self.clone())],
            op: Operation::Tanh,
            ..Default::default()
        })
    }

    fn exp(&self) -> Rc<Self> {
        let x = *self.data.borrow();
        let e = f64::exp(x);
        println!("exp({}) = {}", x, e);
        Rc::new(Self {
            id: Uuid::new_v4(),
            data: RefCell::new(e),
            children: vec![Rc::new(self.clone())],
            op: Operation::Exp,
            ..Default::default()
        })
    }

    fn pow(&self, x: &Value) -> Rc<Self> {
        Rc::new(Self {
            data: RefCell::new(f64::powf(*self.data.borrow(), *x.data.borrow())),
            children: vec![Rc::new(self.clone()), Rc::new(x.clone())],
            op: Operation::Pow,
            ..Default::default()
        })
    }

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
                let op_id = format!("{}{}", node_id, node.op.as_str());
                let lhs_id = &*node.children[0] as *const _ as usize;

                out += &format!(
                    "  \"{}\" [label=\"{}\"]\n",
                    op_id,
                    node.op.as_str().to_string()
                );
                out += &format!("  \"{}\" -> \"{}\"\n", op_id, node_id,);

                out += &format!("  \"{}\" -> \"{}\"\n", lhs_id, op_id,);
                if &node.children.len() == &2 {
                    let rhs_id = &*node.children[1] as *const _ as usize;
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

impl Clone for Value {
    fn clone(&self) -> Self {
        Value {
            id: Uuid::new_v4(),
            label: self.label.clone(),
            data: RefCell::new(*self.data.borrow()),
            children: self.children.clone(),
            op: self.op.clone(),
            grad: RefCell::new(*self.grad.borrow()),
        }
    }
}

impl<'a> Add for &'a Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() + *other.data.borrow(),
            vec![Rc::new(self.clone()), Rc::new(other.clone())],
            Operation::Add,
        )
    }
}

impl<'a> Sub for &'a Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() - *other.data.borrow(),
            vec![Rc::new(self.clone()), Rc::new(other.clone())],
            Operation::Sub,
        )
    }
}

impl<'a> Mul for &'a Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() * *other.data.borrow(),
            vec![Rc::new(self.clone()), Rc::new(other.clone())],
            Operation::Mul,
        )
    }
}

impl<'a> Div for &'a Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() / *other.data.borrow(),
            vec![Rc::new(self.clone()), Rc::new(other.clone())],
            Operation::Div,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_add() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: Some("b".to_string()),
            data: RefCell::new(2.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: RefCell::new(0.0),
        });
        let c = &*a + &*b;
        assert_eq!(*c.data.borrow(), 3.0);
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(c.children[1].data, b.data);

        // Is should be possible to mutate a and b without affecting c
        *a.data.borrow_mut() += 8.0;
        *b.data.borrow_mut() += -8.0;
        assert_eq!(*c.data.borrow(), 3.0);
        assert_eq!(*c.children[0].data.borrow(), 1.0);
        assert_eq!(*c.children[1].data.borrow(), 2.0);
    }

    #[test]
    fn test_add_rhs_f64() {
        let a = Rc::new(Value::new(1.0));
        let c = Value::add_f64(a.clone(), 3.0);
        assert_eq!(*c.data.borrow(), 4.0);
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(*c.children[1].data.borrow(), 3.0);
    }

    #[test]
    fn test_toplogical_order() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value::new(2.0));
        let c = &*a + &*b;
        let order = Value::topological_order(Rc::new(c));
        assert_eq!(order.len(), 3);
        assert_eq!(*order[0].data.borrow(), 3.0);
        assert_eq!(*order[1].data.borrow(), 2.0);
        assert_eq!(*order[2].data.borrow(), 1.0);
    }

    #[test]
    fn test_sub() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: Some("b".to_string()),
            data: RefCell::new(4.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: RefCell::new(0.0),
        });
        let c = &*a - &*b;
        assert_eq!(*c.data.borrow(), -3.0);
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(c.children[1].data, b.data);
    }

    #[test]
    fn test_mul() {
        let a = Rc::new(Value::new(2.0));
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: Some("b".to_string()),
            data: RefCell::new(4.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: RefCell::new(0.0),
        });
        let c = &*a * &*b;
        assert_eq!(*c.data.borrow(), 8.0);
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(c.children[1].data, b.data);
    }

    #[test]
    fn test_div() {
        let a = Rc::new(Value::new(8.0));
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: Some("b".to_string()),
            data: RefCell::new(2.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: RefCell::new(0.0),
        });
        let c = &*a / &*b;
        assert_eq!(*c.data.borrow(), 4.0);
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(c.children[1].data, b.data);
    }

    #[test]
    fn test_backwards() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value::new(2.0));
        let c = &*a + &*b;
        let c = Rc::new(c);
        Value::backwards(c.clone());
        assert_eq!(*c.grad.borrow(), 1.0);
        assert_eq!(*c.children[0].grad.borrow(), 1.0);
        assert_eq!(*c.children[1].grad.borrow(), 1.0);
    }

    #[test]
    fn test_tanh() {
        let x1 = Rc::new(Value::new_with_label(2.0, "x1"));
        let x2 = Rc::new(Value::new_with_label(0.0, "x2"));
        let w1 = Rc::new(Value::new_with_label(-3.0, "w1"));
        let w2 = Rc::new(Value::new_with_label(1.0, "w2"));
        let b = Rc::new(Value::new_with_label(6.8813735870195432, "b"));
        let x1w1 = Rc::new(&*x1 * &*w1);
        //x1w1.set_label("x1*w1");
        let x2w2 = Rc::new(&*x2 * &*w2);
        //x2w2.set_label("x2*w2");
        let x1w1x2w2 = Rc::new(&*x1w1 + &*x2w2);
        //x1w1x2w2.set_label("x1w1x + 2w2");
        let n = Rc::new(&*x1w1x2w2 + &*b);
        //n.set_label("n");
        let o = n.tanh();
        //o.set_label("o");
        Value::backwards(o.clone());

        let order = Value::topological_order(o.clone());
        for (i, v) in order.iter().enumerate() {
            println!("{}: {:?}: {}", i, v.label, *v.grad.borrow());
        }
        assert_abs_diff_eq!(*order[0].grad.borrow(), 1.0);
        assert_abs_diff_eq!(*order[1].grad.borrow(), 0.5);

        assert_abs_diff_eq!(*order[2].grad.borrow(), 0.5);
        assert_abs_diff_eq!(*order[3].grad.borrow(), 0.5);

        assert_abs_diff_eq!(*order[4].grad.borrow(), 0.5);
        assert_abs_diff_eq!(*order[5].grad.borrow(), 0.0);

        assert_abs_diff_eq!(*order[6].grad.borrow(), 0.5);
        assert_abs_diff_eq!(*order[7].grad.borrow(), 0.5);

        assert_abs_diff_eq!(*order[8].grad.borrow(), 1.0);
        assert_abs_diff_eq!(*order[9].grad.borrow(), -1.5, epsilon = 1e-1);
        println!("o.dot: {}", o.dot());
    }

    #[test]
    fn test_exp() {
        let a = Rc::new(Value::new(2.0));
        let c = &*a.exp();
        assert_eq!(*c.data.borrow(), f64::exp(2.0));
        assert_eq!(c.children[0].data, a.data);
    }

    #[test]
    fn test_pow() {
        let a = Rc::new(Value::new(2.0));
        let x = Rc::new(Value::new(3.0));
        let c = &*a.pow(&*x);
        assert_eq!(*c.data.borrow(), f64::powf(2.0, 3.0));
        assert_eq!(c.children[0].data, a.data);
        assert_eq!(c.children[1].data, x.data);
    }
}

/*
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Value {
    id: Uuid,
    data: RefCell<f64>,
    label: Option<String>,
    children: Vec<Rc<Value>>,
    operation: Option<Operation>,
    grad: RefCell<f64>,
}

impl<'a> PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

// Implement Eq for Value
impl<'a> Eq for Value {}

use std::hash::{Hash, Hasher};
impl<'a> Hash for Value {
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
impl Value {
    fn new(data: f64) -> Box<Self> {
        Box::new(Value {
            id: Uuid::new_v4(),
            data: RefCell::new(data),
            label: None,
            children: Vec::new(),
            operation: None,
            grad: RefCell::new(0.0), // we initialize the gradient to 0.0
        })
    }
    fn new_with_label(data: f64, label: &str) -> Box<Self> {
        Box::new(Value {
            id: Uuid::new_v4(),
            data: RefCell::new(data),
            label: Some(label.to_string()),
            children: Vec::new(),
            operation: None,
            grad: RefCell::new(0.0), // we initialize the gradient to 0.0
        })
    }
}

impl Value {
    fn backward(&self) {
        match self.operation {
            Some(Operation::Add) => {
                // Think of this as d = c + e
                // Then &self is d and self.children is (c, e), and since
                // addition passes through the gradient we can just set the
                // gradients of the children to the gradient of d.
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                // If we have have a + a then both lhs and rhs will be then
                // same value so we accumulate the gradient.
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Some(Operation::Sub) => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Some(Operation::Mul) => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Div) => {
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Tanh) => {
                let lhs = &self.children[0];
                *lhs.grad.borrow_mut() += 1.0 - self.data.borrow().powf(2.0) * *self.grad.borrow();
            }
            Some(Operation::Exp) => {
                let lhs = &self.children[0];
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
                let lhs = &self.children[0];
                let rhs = &self.children[1];
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
impl Value {
    fn topological_sort(
        root: &Box<Value>,
        visited: &mut HashSet<Uuid>,
        stack: &mut VecDeque<Box<Value>>,
    ) {
        visited.insert(root.id);

        for child in root.children.iter() {
            if !visited.contains(&child.id) {
                //let child = Box::new(child.as_ref());
                Self::topological_sort(child, visited, stack);
            }
        }

        println!("pushing: {:?}", root.grad);
        stack.push_front(root.clone());
    }

    fn topological_order(root: Box<Value>) -> VecDeque<Box<Value>> {
        let mut visited = HashSet::new();
        let mut stack = VecDeque::new();
        Self::topological_sort(&root, &mut visited, &mut stack);
        stack
    }

    fn backwards(root: Box<Value>) {
        *root.grad.borrow_mut() = 1.0;
        // Now lets do the backpropagation using the topological order.
        for boxed_value in &root.children {
            let value: &Value = boxed_value;
            println!("grad: {:?}", value.grad);
        }

        let order = Value::topological_order(root);
        println!("topological order:");
        for (i, node) in order.iter().enumerate() {
            println!("{}: {:?}", i, node.data);
            node.backward();
        }
    }
}

impl Value {
    fn new_with_children(
        data: f64,
        label: Option<String>,
        lhs: Rc<Value>,
        rhs: Option<Rc<Value>>,
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
impl Add<Box<Value>> for Rc<Value> {
    type Output = Value;

    fn add(self, other: Rc<Value>) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() + *other.data.borrow(),
            None,
            self.clone(),
            Some(other.clone()),
            Operation::Add,
        )
    }
}

impl Add<Box<Value>> for f64 {
    type Output = Value;

    fn add(self, other: Box<Value>) -> Self::Output {
        Value::new_with_children(
            self + *other.data.borrow(),
            None,
            Value::new(self),
            Some(other.clone()),
            Operation::Add,
        )
    }
}

impl Add<f64> for Box<Value> {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() + other,
            None,
            self.clone(),
            Some(Value::new(other)),
            Operation::Add,
        )
    }
}

// Add Sub trait implementation for Value and add use statement
use std::ops::Sub;
impl Sub<Value> for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() - *other.data.borrow(),
            None,
            Box::new(self.clone()),
            Some(Box::new(other.clone())),
            Operation::Sub,
        )
    }
}

impl Sub<Value> for f64 {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        Value::new_with_children(
            self - *other.data.borrow(),
            None,
            Box::new(other.clone()),
            Some(Value::new(self)),
            Operation::Sub,
        )
    }
}

impl Sub<f64> for Value {
    type Output = Value;

    fn sub(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() - other,
            None,
            Box::new(self.clone()),
            Some(Value::new(other)),
            Operation::Sub,
        )
    }
}

// Add Mul trait implementation for Value and add use statement
use std::ops::Mul;
impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() * *other.data.borrow(),
            None,
            Box::new(self.clone()),
            Some(Box::new(other.clone())),
            Operation::Mul,
        )
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        Value::new_with_children(
            self * *other.data.borrow(),
            None,
            Value::new(self),
            Some(Box::new(other.clone())),
            Operation::Mul,
        )
        //Value::new(self * *other.data.borrow())
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() * other,
            None,
            Box::new(self.clone()),
            Some(Value::new(other)),
            Operation::Mul,
        )
    }
}

use std::ops::Div;
impl Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() / *other.data.borrow(),
            None,
            Box::new(self.clone()),
            Some(Box::new(other.clone())),
            Operation::Div,
        )
    }
}

impl Div<Value> for f64 {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        Value::new_with_children(
            self / *other.data.borrow(),
            None,
            Value::new(self),
            Some(Box::new(other.clone())),
            Operation::Div,
        )
    }
}

impl Div<f64> for Value {
    type Output = Value;

    fn div(self, other: f64) -> Self::Output {
        Value::new_with_children(
            *self.data.borrow() / other,
            None,
            Box::new(self.clone()),
            Some(Value::new(other)),
            Operation::Div,
        )
    }
}

// Implement the Display trait for Value in the format Value(data) and
// include any necessary use statements
use std::fmt;
impl fmt::Display for Value {
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
impl Value {
    fn operation(&self) -> Option<Operation> {
        self.operation.clone()
    }

    fn label(&mut self, label: &str) {
        self.label = Some(label.to_string())
    }

    fn grad(&self) -> f64 {
        *self.grad.borrow()
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
        Value::new_with_children(t, None, Box::new(self.clone()), None, Operation::Tanh)
    }

    fn exp(&self) -> Value {
        let x = *self.data.borrow();
        let e = f64::exp(x);
        println!("exp({}) = {}", x, e);
        Value::new_with_children(e, None, Box::new(self.clone()), None, Operation::Exp)
    }

    fn pow(&self, x: Value) -> Value {
        println!("pow({}, {})", *self.data.borrow(), x);
        Value::new_with_children(
            f64::powf(*self.data.borrow(), *x.data.borrow()),
            None,
            Box::new(self.clone()),
            Some(Box::new(x.clone())),
            Operation::Pow,
        )
    }
}

#[allow(dead_code)]
impl Value {
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
                let lhs_id = &*node.children[0] as *const _ as usize;

                out += &format!(
                    "  \"{}\" [label=\"{}\"]\n",
                    op_id,
                    node.operation.as_ref().unwrap().as_str().to_string()
                );
                out += &format!("  \"{}\" -> \"{}\"\n", op_id, node_id,);

                out += &format!("  \"{}\" -> \"{}\"\n", lhs_id, op_id,);
                if &node.children.len() == &2 {
                    let rhs_id = &*node.children[1] as *const _ as usize;
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
    let c = a + b;
    println!("c = {}", c);

    assert_eq!(*c.data.borrow(), 3.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_rhs_float() {
    let a = Value::new(1.0);
    let c = a + 4.0;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(*c.children[1].data.borrow(), 4.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_lhs_float() {
    let a = Value::new(1.0);
    let c = 4.0 + a;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 4.0);
    assert_eq!(*c.children[1].data.borrow(), 1.0);
    assert_eq!(c.operation, Some(Operation::Add));
}

#[test]
fn test_add_backwards() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = a + b;
    Value::backwards(Box::new(c.clone()));

    //assert_eq!(c.grad(), 1.0);
    println!("c.children[0].grad() = {}", c.children[0].grad());
    println!("c.children[1].grad() = {}", c.children[1].grad());
    //assert_eq!(a.grad(), 1.0);
    //assert_eq!(*b.grad.borrow(), 1.0);
}

// Add a test to test subtraction
#[test]
fn test_sub() {
    let a = Value::new(1.0);
    let b = Value::new(2.0);
    let c = a - b;

    assert_eq!(*c.data.borrow(), -1.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 1.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Sub));
}

#[test]
fn test_sub_lhs_float() {
    let a = Value::new(8.0);
    let c = 4.0 - a;

    assert_eq!(*c.data.borrow(), -4.0);
    assert_eq!(c.operation, Some(Operation::Sub));
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 8.0);
}

#[test]
fn test_sub_rhs_float() {
    let a = Value::new(8.0);
    let c = a - 4.0;

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
    let c = a / b;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 2);
    assert_eq!(*c.children[0].data.borrow(), 10.0);
    assert_eq!(*c.children[1].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Div));
}

#[test]
fn test_div_lhs_float() {
    let a = Value::new(2.0);
    let c = 10.0 / a;

    assert_eq!(*c.data.borrow(), 5.0);
    assert_eq!(c.children.len(), 1);
    assert_eq!(*c.children[0].data.borrow(), 2.0);
    assert_eq!(c.operation, Some(Operation::Div));
}

#[test]
fn test_div_rhs_float() {
    let a = Value::new(10.0);
    let c = a / 2.0;

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
    let mut x1w1 = x1 * w1;
    x1w1.label("x1*w1");
    let mut x2w2 = x2 * w2;
    x2w2.label("x2*w2");
    let mut x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.label("x1w1x + 2w2");
    let mut n = x1w1x2w2 + b;
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
    let mut x1w1 = x1 * w1;
    x1w1.label("x1*w1");
    let mut x2w2 = x2 * w2;
    x2w2.label("x2*w2");
    let mut x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.label("x1w1x + 2w2");
    let mut n = x1w1x2w2 + b;
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
    let e_minus_one = e_two_exp - 1.0;
    let e_plus_one = e_two_exp + 1.0;
    let mut o = e_minus_one / e_plus_one;
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
*/

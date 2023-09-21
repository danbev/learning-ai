use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use uuid::Uuid;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Value {
    pub id: Uuid,
    pub label: RefCell<String>,
    pub data: RefCell<f64>,
    pub children: Vec<Rc<Value>>,
    pub op: Operation,
    pub grad: Rc<RefCell<f64>>,
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
            label: RefCell::new("N/A".to_string()),
            data: RefCell::new(0.0),
            children: vec![],
            op: Operation::None,
            grad: Rc::new(RefCell::new(0.0)),
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
            label: RefCell::new(label.to_string()),
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

    pub fn label(&self) -> RefCell<String> {
        self.label.clone()
    }

    pub fn set_label(&mut self, label: &str) {
        *self.label.borrow_mut() = label.to_string();
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

    pub fn backward(&self) {
        match self.op {
            Operation::Add => {
                // Think of this as c = a + b
                // And we have the derivative function def:
                // f(x+h) - f(x)
                // -------------
                //      h
                // So we plug in our function which is f(a + b) and we get:
                // df/da = (a + h) + b) - (a + b) / h
                // df/da = a + h + b - a - b / h
                // df/da = X + h + b - X - b / h
                // df/da = h + b - b / h
                // df/da = h + X - X / h
                // df/da = h / h = 1
                // So the derivative of a is 1.
                //
                // df/db = (a) + (b + h) - (a + b) / h
                // df/db = a + b + h - a - b / h
                // df/db = a + h + X - a - X / h
                // df/db = a + h - a / h
                // df/db = X + h - X / h
                // df/db = h / h = 1
                // So the derivative of b is also 1.
                //
                // &self is c and self.children are (a, b).
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                // If we have have a + a then both lhs and rhs will be point
                // to the same value, so we accumulate the gradient.
                // The multiplication is the chain rule.
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
            }
            Operation::Sub => {
                // Think of this as c = a + b
                // And we have the derivative function def:
                // f(x+h) - f(x)
                // -------------
                //      h
                // So we plug in our function which is f(a - b) and we get:
                // df/da = (a + h) - b) - (a - b) / h
                // df/da = a + h - b - a + b) / h
                // df/da = X + h - b - X + b) / h
                // df/da = h - b + b) / h
                // df/da = h / h = 1
                //
                // df/db = (a - (b + h)) - (a - b) / h
                // df/db = a - b - h - a + b / h
                // df/db = X - b - h - X + b / h
                // df/db = - b - h + b / h
                // df/db = - X - h + X / h
                // df/db = -h/h = -1
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += 1.0 * *self.grad.borrow();
                *rhs.grad.borrow_mut() -= 1.0 * *self.grad.borrow();
            }
            Operation::Mul => {
                // Think of this as c = a * b
                // And we have the derivative function def:
                // f(x+h) - f(x)
                // -------------
                //      h
                // So we plug in our function which is f(a * b) and we get:
                // df/da = (a + h) * b) - (a * b) / h
                // df/da = ab + hb - ab / h
                // df/da = X  + hb - X / h
                // df/da = hb / h
                // df/da = Xb / X
                // df/da = b
                //
                // df/db = (a * (b + h) - (a * b) / h
                // df/db = ab + ah - ab / h
                // df/db = X + ah - X / h
                // df/db = ah / h
                // df/db = aX / X
                // df/db = a
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                *lhs.grad.borrow_mut() += *rhs.data.borrow() * *self.grad.borrow();
                *rhs.grad.borrow_mut() += *lhs.data.borrow() * *self.grad.borrow();
            }
            Operation::Div => {
                // Think of this as c = a / b
                // And we have the derivative function def:
                // f(x+h) - f(x)
                // -------------
                //      h
                // So we plug in our function which is f(a / b) and we get:
                // df/da = (a + h) / b) - (a / b) / h
                // df/da = a/b + h/b - a/b / h
                // df/da =  X  + h/b -  X / h
                // df/da = h/b / h
                // df/da = h/b * 1/h
                // df/da =  h    1
                //         -- * -- = 1/b
                //          b    h
                // df/da =  X    1    1
                //         -- * -- =  -
                //          b    X    b
                // So df/da = 1/b

                // And now we do the same for b:
                // df/db = a / (b + h) - (a / b) / h
                // df/db = a / (b + h) - (a / b) / h
                //          a       a   ab - a(b+h)   ab - ab -ah
                //         ----- -  - = ----------- = -----------
                //         b + h    b   (b + h)b       (b + h)b
                //
                //                                     X - X -ah
                //                                  = -----------
                //                                     (b + h)bh
                //
                //                                       -aX
                //                                  = -----------
                //                                     (b + h)bX
                //
                //                                       -a
                //                                  = -----------
                //                                     b² + bh
                // As h aproaches 0, the denominator aproaches:
                //                                       -a
                //                                  = -----------
                //                                        b² + 0
                // So df/db = -a / b²
                let lhs = &self.children[0];
                let rhs = &self.children[1];
                // Gradient for 'a' in 'a / b'
                *lhs.grad.borrow_mut() += 1.0 / *rhs.data.borrow() * *self.grad.borrow();
                // Gradient for 'b' in 'a / b'
                *rhs.grad.borrow_mut() -= *lhs.data.borrow()
                    / (*rhs.data.borrow() * *rhs.data.borrow())
                    * *self.grad.borrow();
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
                println!("----------->self: {}", *self.data.borrow());
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

    pub fn tanh(&self) -> Rc<Self> {
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

    pub fn exp(&self) -> Rc<Self> {
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

    pub fn pow(&self, x: &Value) -> Rc<Self> {
        Rc::new(Self {
            data: RefCell::new(f64::powf(*self.data.borrow(), *x.data.borrow())),
            children: vec![Rc::new(self.clone()), Rc::new(x.clone())],
            op: Operation::Pow,
            ..Default::default()
        })
    }

    pub fn dot(&self) -> String {
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

            out += &format!(
                "  \"{}\" [label=\"{} value: {:.4}, grad: {:.4}\" shape=record]\n",
                node_ptr as usize,
                &*node.label.borrow(),
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
            id: self.id,
            label: self.label.clone(),
            data: RefCell::new(*self.data.borrow()),
            children: self.children.clone(),
            op: self.op.clone(),
            grad: self.grad.clone(),
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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Value(data={}, label: {}",
            *self.data.borrow(),
            *self.label.borrow()
        )?;
        if &self.children.len() > &0 {
            write!(f, ", lhs={}", *self.children[0].data.borrow())?;
            if &self.children.len() == &2 {
                write!(f, ", rhs={}", *self.children[1].data.borrow())?;
            }
            write!(f, ", op=\"{:?}\"", &self.op)?;
        }
        write!(f, ", grad={})", self.grad.borrow())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_add() {
        let a = Rc::new(Value::new(1.0));
        *a.label.borrow_mut() = "a".to_string();
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: RefCell::new("b".to_string()),
            data: RefCell::new(2.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: Rc::new(RefCell::new(0.0)),
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
    fn test_toplogical_order_grad() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value::new(2.0));

        let c = Rc::new(&*a + &*b);

        *c.grad.borrow_mut() = 1.0;
        *a.grad.borrow_mut() = 11.0;
        *b.grad.borrow_mut() = 0.5;

        assert_eq!(*c.children[0].grad.borrow(), 11.0);
        assert_eq!(*c.children[1].grad.borrow(), 0.5);
    }

    #[test]
    fn test_sub() {
        let a = Rc::new(Value::new(1.0));
        let b = Rc::new(Value {
            id: Uuid::new_v4(),
            label: RefCell::new("b".to_string()),
            data: RefCell::new(4.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: Rc::new(RefCell::new(0.0)),
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
            label: RefCell::new("b".to_string()),
            data: RefCell::new(4.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: Rc::new(RefCell::new(0.0)),
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
            label: RefCell::new("b".to_string()),
            data: RefCell::new(2.0),
            children: vec![a.clone()],
            op: Operation::None,
            grad: Rc::new(RefCell::new(0.0)),
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
    fn test_tanh_decomposed() {
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

        // Here we want to use the following formula for deriving tanh:
        //         e²ˣ - 1
        // tanh =  ----------
        //         e²ˣ + 1
        //
        //let binding = &n * 2.0;
        //let e = binding.exp();
        //println!("e.children[0]: {}", &e.children[0]);
        println!("n--->: {}", n);
        let e_two_exp = &*n * &Rc::new(Value::new(2.0));
        println!("e_two_exp: {}", e_two_exp);
        let e_two_exp = e_two_exp.exp();
        println!("e_two_exp: {}", e_two_exp);
        let e_minus_one = &*e_two_exp - &Rc::new(Value::new(1.0));
        println!("e_minus_one: {}", e_minus_one);
        let e_plus_one = &*e_two_exp + &Rc::new(Value::new(1.0));
        println!("e_plus_one: {}", e_plus_one);
        let o = &*Rc::new(e_minus_one) / &Rc::new(e_plus_one);
        let o = Rc::new(o.clone());
        println!("o: {}", o);
        //o.label("o");

        //let o = n.tanh();
        //o.set_label("o");
        Value::backwards(o.clone());

        let order = Value::topological_order(o.clone());
        for (i, v) in order.iter().enumerate() {
            println!("{}: {:?}: {}", i, v.label, *v.grad.borrow());
        }
        /*
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
        */
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

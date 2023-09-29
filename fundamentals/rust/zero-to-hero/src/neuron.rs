use crate::value::Value;
use rand::Rng;
use std::rc::Rc;

/// This struct is the equivalent of the `Neuron` class in the python version.
/// which is defined in part1.
/// It represents a single neuron in a layer of allow neural network
#[allow(dead_code)]
#[derive(Debug)]
pub struct Neuron<const I: usize> {
    bias: Rc<Value>,
    weights: [Rc<Value>; I],
}

#[allow(dead_code)]
impl<const I: usize> Neuron<I> {
    pub fn new() -> Neuron<I> {
        let mut rng = rand::thread_rng();
        let weights: Vec<Rc<Value>> = (0..I)
            .map(|i| {
                Rc::new(Value::new_with_label(
                    rng.gen_range(-1.0..=1.0),
                    &format!("w{i}"),
                ))
            })
            .collect();

        Neuron {
            bias: Rc::new(Value::new(rng.gen_range(-1.0..=1.0))),
            weights: weights.try_into().unwrap(),
        }
    }
}

impl<const I: usize> Fn<([Rc<Value>; I],)> for Neuron<I> {
    extern "rust-call" fn call(&self, _args: ([Rc<Value>; I],)) -> Self::Output {
        let mut sum = Value::new_with_label(*self.bias.data.borrow(), "sum");
        for (x, w) in self.weights.iter().zip(_args.0.iter()) {
            sum = &sum + &(&**w * &**x);
        }
        sum.tanh()
    }
}

impl<const I: usize> FnMut<([Rc<Value>; I],)> for Neuron<I> {
    extern "rust-call" fn call_mut(&mut self, _args: ([Rc<Value>; I],)) -> Self::Output {
        self.call(_args)
    }
}

#[allow(dead_code)]
impl<const I: usize> FnOnce<([Rc<Value>; I],)> for Neuron<I> {
    type Output = Rc<Value>;

    /// This function performs the operation of the neuron, that is it
    /// calculates the weighted sum of the inputs plus the bias.
    ///
    /// This is the equivalent of the `__call__` method used in the python
    /// version.
    extern "rust-call" fn call_once(self, _args: ([Rc<Value>; I],)) -> Self::Output {
        self.call(_args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron() {
        let n = Neuron::<3>::new();
        assert!(*n.bias.data.borrow() > -1.0 && *n.bias.data.borrow() < 1.0);
        assert_eq!(n.weights.len(), 3);
    }

    #[test]
    fn test_neuron_call() {
        let n = Neuron::<6>::new();
        let r = n([
            Rc::new(Value::new_with_label(1.0, "x0")),
            Rc::new(Value::new_with_label(2.0, "x1")),
            Rc::new(Value::new_with_label(3.0, "x2")),
            Rc::new(Value::new_with_label(4.0, "x3")),
            Rc::new(Value::new_with_label(5.0, "x4")),
            Rc::new(Value::new_with_label(6.0, "x5")),
        ]);
        println!("r = {}", r);
        println!("r = {}", r.dot());
        let r = n([
            Rc::new(Value::new_with_label(1.0, "x0")),
            Rc::new(Value::new_with_label(2.0, "x1")),
            Rc::new(Value::new_with_label(3.0, "x2")),
            Rc::new(Value::new_with_label(4.0, "x3")),
            Rc::new(Value::new_with_label(5.0, "x4")),
            Rc::new(Value::new_with_label(6.0, "x5")),
        ]);
        println!("r = {}", r);
    }
}

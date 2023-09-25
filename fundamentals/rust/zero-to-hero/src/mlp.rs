use crate::layer::Layer;
use crate::value::Value;
use std::rc::Rc;

// Represents a multi layer perceptron.
//
#[derive(Debug)]
pub struct Mlp<const I: usize, const N: usize, const L: usize> {
    first: Layer<I, N>,
    layers: Vec<Layer<N, N>>,
}

impl<const I: usize, const N: usize, const L: usize> Mlp<I, N, L> {
    pub fn new() -> Mlp<I, N, L> {
        Self {
            first: Layer::<I, N>::new(),
            layers: (1..L).map(|_| Layer::<N, N>::new()).collect(),
        }
    }

    pub fn nr_of_layers(&self) -> usize {
        L
    }

    pub fn nr_of_inputs(&self) -> usize {
        I
    }

    pub fn nr_of_neurons(&self) -> usize {
        N
    }
}

#[allow(dead_code)]
impl<const I: usize, const N: usize, const L: usize> FnOnce<([Rc<Value>; I],)> for Mlp<I, N, L> {
    type Output = Rc<Value>;

    extern "rust-call" fn call_once(self, xs: ([Rc<Value>; I],)) -> Self::Output {
        // The output of one layer is the input to the next layer.
        let mut outputs = (self.first)(xs.0);
        for (i, layer) in self.layers.into_iter().enumerate() {
            if i == 0 {
            } else {
                outputs = layer(outputs);
            }
        }
        let last = Layer::<N, 1>::new();
        last(outputs)[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp() {
        let l = Mlp::<1, 3, 2>::new();
        assert_eq!(l.nr_of_inputs(), 1);
        assert_eq!(l.nr_of_neurons(), 3);
        assert_eq!(l.nr_of_layers(), 2);
    }

    #[test]
    fn test_mlp_call() {
        let layer = Mlp::<1, 3, 2>::new();
        let inputs = [Rc::new(Value::new_with_label(1.0, "x0"))];
        let output = layer(inputs);
        println!("output: {}", output.dot());
    }
}

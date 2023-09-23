use crate::layer::Layer;
use crate::value::Value;
use std::rc::Rc;

// Represents a multi layer perceptron.
//
#[derive(Debug)]
pub struct Mlp<const I: usize, const N: usize, const L: usize> {
    layers: [Layer<I, N>; L],
}

impl<const I: usize, const N: usize, const L: usize> Mlp<I, N, L> {
    pub fn new() -> Mlp<I, N, L> {
        let layers: Vec<Layer<I, N>> = (0..L).map(|_| Layer::<I, N>::new()).collect();
        Self {
            layers: layers.try_into().unwrap(),
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
impl<const I: usize, const N: usize, const L: usize> FnOnce<(Vec<Rc<Value>>,)> for Mlp<I, N, L> {
    type Output = Vec<Rc<Value>>;

    extern "rust-call" fn call_once(self, xs: (Vec<Rc<Value>>,)) -> Self::Output {
        // The output of one layer is the input to the next layer.
        let mut layer_output = xs.0;
        for layer in self.layers.into_iter() {
            layer_output = layer(layer_output);
        }
        layer_output
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
        let inputs = vec![Rc::new(Value::new_with_label(1.0, "x0"))];
        let outputs = layer(inputs);
        assert_eq!(outputs.len(), 3);
    }
}

use crate::neuron::Neuron;
use crate::value::Value;
use std::rc::Rc;

// Represents a single layer in a neural network.
// A layer has one or more neurons.
//
#[derive(Debug)]
pub struct Layer<const I: usize, const N: usize> {
    neurons: Vec<Neuron<I>>,
}

impl<const I: usize, const N: usize> Layer<I, N> {
    pub fn new() -> Layer<I, N> {
        Layer {
            neurons: (0..N).map(|_| Neuron::<I>::new()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn inputs_len(&self) -> usize {
        I
    }
}

impl<const I: usize, const N: usize> Fn<([Rc<Value>; I],)> for Layer<I, N> {
    extern "rust-call" fn call(&self, args: ([Rc<Value>; I],)) -> Self::Output {
        let outputs = self
            .neurons
            .iter()
            .map(|neuron| neuron(args.0.clone()))
            .collect::<Vec<Rc<Value>>>();
        outputs.try_into().unwrap()
    }
}

impl<const I: usize, const N: usize> FnMut<([Rc<Value>; I],)> for Layer<I, N> {
    extern "rust-call" fn call_mut(&mut self, args: ([Rc<Value>; I],)) -> Self::Output {
        self.call(args)
    }
}

#[allow(dead_code)]
impl<const I: usize, const N: usize> FnOnce<([Rc<Value>; I],)> for Layer<I, N> {
    type Output = [Rc<Value>; N];

    extern "rust-call" fn call_once(self, args: ([Rc<Value>; I],)) -> Self::Output {
        self.call(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer() {
        let l = Layer::<1, 3>::new();
        assert_eq!(l.len(), 3);
        assert_eq!(l.inputs_len(), 1);
    }

    #[test]
    fn test_layer_call() {
        let inputs = [
            Rc::new(Value::new_with_label(1.0, "x0")),
            Rc::new(Value::new_with_label(2.0, "x1")),
        ];
        let layer = Layer::<2, 3>::new();
        assert_eq!(layer.len(), 3);
        assert_eq!(layer.inputs_len(), 2);

        let outputs = layer(inputs);
        assert_eq!(outputs.len(), 3);

        let inputs = [
            Rc::new(Value::new_with_label(1.0, "x0")),
            Rc::new(Value::new_with_label(2.0, "x1")),
        ];
        let outputs = layer(inputs);
        assert_eq!(outputs.len(), 3);
    }
}

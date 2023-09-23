use crate::neuron::Neuron;
use crate::value::Value;
use std::rc::Rc;

// Represents a single layer in a neural network.
// A layer has one or more neurons.
//
#[derive(Debug)]
pub struct Layer<const I: usize, const N: usize> {
    neurons: [Neuron<I>; N],
}

impl<const I: usize, const N: usize> Layer<I, N> {
    pub fn new() -> Layer<I, N> {
        let neurons: Vec<Neuron<I>> = (0..N).map(|_| Neuron::<I>::new()).collect();
        Layer {
            neurons: neurons.try_into().unwrap(),
        }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn inputs_len(&self) -> usize {
        I
    }
}

#[allow(dead_code)]
impl<const I: usize, const N: usize> FnOnce<([Rc<Value>; I],)> for Layer<I, N> {
    type Output = [Rc<Value>; N];

    extern "rust-call" fn call_once(self, xs: ([Rc<Value>; I],)) -> Self::Output {
        let outputs = self
            .neurons
            .into_iter()
            .map(|neuron| neuron(xs.0.clone()))
            .collect::<Vec<Rc<Value>>>();
        outputs.try_into().unwrap()
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
    }
}

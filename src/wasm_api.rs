use wasm_bindgen::prelude::*;

use std::collections::VecDeque;
use std::iter::zip;

use crate::model;
use crate::ops::{Input, Output};
use crate::tensor;

#[wasm_bindgen]
pub struct Model {
    model: model::Model,
}

#[wasm_bindgen]
impl Model {
    /// Construct a new model from a serialized graph.
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: &[u8]) -> Result<Model, String> {
        let model = model::load_model(model_data)?;
        Ok(Model { model })
    }

    /// Find the ID of a node in the graph from its name.
    #[wasm_bindgen(js_name = findNode)]
    pub fn find_node(&self, name: &str) -> Option<usize> {
        self.model.find_node(name)
    }

    /// Execute the model, passing `input` as the tensor values for the node
    /// IDs specified by `input_ids` and calculating the values of the nodes
    /// specified by `output_ids`.
    pub fn run(
        &self,
        input_ids: &[usize],
        input: &TensorList,
        output_ids: &[usize],
    ) -> Result<TensorList, String> {
        let inputs: Vec<(usize, Input)> = zip(
            input_ids.iter().copied(),
            input.tensors.iter().map(|tensor| (&tensor.data).into()),
        )
        .collect();
        let result = self.model.run(&inputs[..], output_ids, None);
        match result {
            Ok(outputs) => {
                let mut list = TensorList::new();
                for output in outputs.into_iter() {
                    list.push(Tensor::from_output(output));
                }
                Ok(list)
            }
            Err(err) => Err(format!("{:?}", err)),
        }
    }
}

/// A wrapper around a multi-dimensional array model input or output.
#[wasm_bindgen]
pub struct Tensor {
    data: Output,
}

#[wasm_bindgen]
impl Tensor {
    #[wasm_bindgen(js_name = floatTensor)]
    pub fn float_tensor(shape: &[usize], data: &[f32]) -> Tensor {
        let data: Output = tensor::Tensor::from_data(shape.into(), data.into()).into();
        Tensor { data }
    }

    #[wasm_bindgen(js_name = intTensor)]
    pub fn int_tensor(shape: &[usize], data: &[i32]) -> Tensor {
        let data: Output = tensor::Tensor::from_data(shape.into(), data.into()).into();
        Tensor { data }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self.data {
            Output::IntTensor(ref t) => t.shape().into(),
            Output::FloatTensor(ref t) => t.shape().into(),
        }
    }

    #[wasm_bindgen(js_name = floatData)]
    pub fn float_data(&self) -> Option<Vec<f32>> {
        match self.data {
            Output::FloatTensor(ref t) => Some(t.elements_vec()),
            _ => None,
        }
    }

    #[wasm_bindgen(js_name = intData)]
    pub fn int_data(&self) -> Option<Vec<i32>> {
        match self.data {
            Output::IntTensor(ref t) => Some(t.elements_vec()),
            _ => None,
        }
    }

    fn from_output(out: Output) -> Tensor {
        Tensor { data: out }
    }
}

/// A list of tensors that can be passed as the input to or received as the
/// result from a model run.
///
/// Due to wasm-bindgen constraints, this structure has a queue-like interface
/// that only supports adding and removing items, but not retrieving a reference
/// to an item at an arbitrary index. JS code will likely want to convert this
/// into a JS array for more convenient access.
#[wasm_bindgen]
pub struct TensorList {
    tensors: VecDeque<Tensor>,
}

#[wasm_bindgen]
impl TensorList {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TensorList {
        TensorList {
            tensors: VecDeque::new(),
        }
    }

    /// Add a new tensor to the end of the list.
    pub fn push(&mut self, tensor: Tensor) {
        self.tensors.push_back(tensor);
    }

    /// Remove and return the first tensor from this list.
    ///
    /// This method is named `shift` as it matches the behavior of
    /// `Array::shift` in JS.
    #[wasm_bindgen(js_name = shift)]
    pub fn shift(&mut self) -> Option<Tensor> {
        self.tensors.pop_front()
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.tensors.len()
    }
}

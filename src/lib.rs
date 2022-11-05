use std::iter::zip;

use wasm_bindgen::prelude::*;

mod graph;
mod linalg;
mod model;
mod ops;
mod rng;
mod tensor;
mod timer;

pub use graph::RunOptions;
pub use model::load_model;
pub use tensor::{from_data, zero_tensor, Tensor};

#[allow(dead_code, unused_imports)]
mod schema_generated;

#[cfg(test)]
mod model_builder;

#[cfg(test)]
mod test_util;

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
    pub fn run(&self, input_ids: &[usize], input: TensorList, output_ids: &[usize]) -> TensorList {
        let inputs: Vec<_> = zip(input_ids.iter().copied(), input.tensors.iter()).collect();
        let outputs = self
            .model
            .run(&inputs[..], output_ids, None)
            .into_iter()
            .map(|out| out.as_float().unwrap())
            .collect();
        TensorList::from_vec(outputs)
    }
}

/// A list of tensors that can be passed as the input to or received as the
/// result from a model run.
#[wasm_bindgen]
pub struct TensorList {
    tensors: Vec<Tensor>,
}

#[wasm_bindgen]
impl TensorList {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TensorList {
        TensorList {
            tensors: Vec::new(),
        }
    }

    /// Add a new tensor to the list with the given shape and data.
    pub fn push(&mut self, shape: &[usize], data: &[f32]) {
        self.tensors.push(from_data(shape.into(), data.into()));
    }

    /// Extract the dimensions of a tensor in the list.
    #[wasm_bindgen(js_name = getShape)]
    pub fn get_shape(&self, index: usize) -> Option<Vec<usize>> {
        self.tensors.get(index).map(|t| t.shape().into())
    }

    /// Extract the elements of a tensor in the list.
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self, index: usize) -> Option<Vec<f32>> {
        self.tensors.get(index).map(|t| t.data().into())
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.tensors.len()
    }

    fn from_vec(tensors: Vec<Tensor>) -> TensorList {
        TensorList { tensors }
    }
}

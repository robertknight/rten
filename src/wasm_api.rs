use wasm_bindgen::prelude::*;

use std::iter::zip;
use std::rc::Rc;

use wasnn_tensor::TensorLayout;

use crate::graph::Dimension;
use crate::model;
use crate::ops::{Input, Output};

#[wasm_bindgen]
pub struct Model {
    model: model::Model,
}

#[wasm_bindgen]
impl Model {
    /// Construct a new model from a serialized graph.
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: &[u8]) -> Result<Model, String> {
        let model = model::Model::load(model_data)?;
        Ok(Model { model })
    }

    /// Find the ID of a node in the graph from its name.
    #[wasm_bindgen(js_name = findNode)]
    pub fn find_node(&self, name: &str) -> Option<usize> {
        self.model.find_node(name)
    }

    /// Get metadata about the node with a given ID.
    ///
    /// This is useful for getting the input tensor shape expected by the model.
    #[wasm_bindgen(js_name = nodeInfo)]
    pub fn node_info(&self, id: usize) -> Option<NodeInfo> {
        self.model.node_info(id).map(|ni| NodeInfo {
            name: ni.name().map(|n| n.to_string()),
            shape: ni.shape(),
        })
    }

    /// Return the IDs of input nodes.
    ///
    /// Additional details about the nodes can be obtained using `node_info`.
    #[wasm_bindgen(js_name = inputIds)]
    pub fn input_ids(&self) -> Vec<usize> {
        self.model.input_ids().into()
    }

    /// Return the IDs of output nodes.
    ///
    /// Additional details about the nodes can be obtained using `node_info`.
    #[wasm_bindgen(js_name = outputIds)]
    pub fn output_ids(&self) -> Vec<usize> {
        self.model.output_ids().into()
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
            input.tensors.iter().map(|tensor| (&*tensor.data).into()),
        )
        .collect();
        let result = self.model.run(&inputs[..], output_ids, None);
        match result {
            Ok(outputs) => {
                let mut list = TensorList::new();
                for output in outputs.into_iter() {
                    list.push(&Tensor::from_output(output));
                }
                Ok(list)
            }
            Err(err) => Err(format!("{:?}", err)),
        }
    }
}

/// Metadata about a node in the model.
#[wasm_bindgen]
pub struct NodeInfo {
    name: Option<String>,
    shape: Option<Vec<Dimension>>,
}

#[wasm_bindgen]
impl NodeInfo {
    /// Returns the name of a node in the graph, if it has one.
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// Returns the tensor shape of a node in the graph.
    ///
    /// For inputs, this specifies the shape that the model expects the input
    /// to have. Dimensions can be -1 if the model does not specify a size.
    ///
    /// Note: Ideally this would return `null` for unknown dimensions, but
    /// wasm_bindgen does not support returning a `Vec<Option<i32>>`.
    pub fn shape(&self) -> Option<Vec<i32>> {
        self.shape.as_ref().map(|dims| {
            dims.iter()
                .map(|dim| match dim {
                    Dimension::Fixed(size) => *size as i32,
                    Dimension::Symbolic(_) => -1,
                })
                .collect()
        })
    }
}

/// A wrapper around a multi-dimensional array model input or output.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Tensor {
    data: Rc<Output>,
}

#[wasm_bindgen]
impl Tensor {
    /// Construct a float tensor from the given shape and data.
    #[wasm_bindgen(js_name = floatTensor)]
    pub fn float_tensor(shape: &[usize], data: &[f32]) -> Tensor {
        let data: Output = wasnn_tensor::Tensor::from_data(shape, data.to_vec()).into();
        Tensor {
            data: Rc::new(data),
        }
    }

    /// Construct an int tensor from the given shape and data.
    #[wasm_bindgen(js_name = intTensor)]
    pub fn int_tensor(shape: &[usize], data: &[i32]) -> Tensor {
        let data: Output = wasnn_tensor::Tensor::from_data(shape, data.to_vec()).into();
        Tensor {
            data: Rc::new(data),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().into()
    }

    /// Return the elements of a float tensor in their logical order.
    #[wasm_bindgen(js_name = floatData)]
    pub fn float_data(&self) -> Option<Vec<f32>> {
        match *self.data {
            Output::FloatTensor(ref t) => Some(t.to_vec()),
            _ => None,
        }
    }

    /// Return the elements of an int tensor in their logical order.
    #[wasm_bindgen(js_name = intData)]
    pub fn int_data(&self) -> Option<Vec<i32>> {
        match *self.data {
            Output::IntTensor(ref t) => Some(t.to_vec()),
            _ => None,
        }
    }

    fn from_output(out: Output) -> Tensor {
        Tensor { data: Rc::new(out) }
    }
}

/// A list of tensors that can be passed as the input to or received as the
/// result from a model run.
///
/// This custom list class exists because wasm-bindgen does not support passing
/// or returning arrays of custom structs. The interface of this class is
/// similar to array-like DOM APIs like `NodeList`. Like `NodeList`, TensorList
/// is iterable and can be converted to an array using `Array.from` (nb. the
/// iterator implementation is defined in JS).
#[wasm_bindgen]
#[derive(Default)]
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

    /// Add a new tensor to the end of the list.
    pub fn push(&mut self, tensor: &Tensor) {
        self.tensors.push(tensor.clone());
    }

    /// Return the item at a given index.
    pub fn item(&self, index: usize) -> Option<Tensor> {
        self.tensors.get(index).cloned()
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.tensors.len()
    }
}

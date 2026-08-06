//! The `Model` class and the metadata types it exposes.

use std::path::PathBuf;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rten::{Dimension, NodeId, ValueOrView};

use crate::RtenError;
use crate::value::{input_tensor, value_to_py};

/// A machine learning model, loaded from a `.onnx` or `.rten` file.
///
/// Models are loaded by passing a file path or the contents of a model file:
///
/// ```python
/// model = rten.Model("model.onnx")
/// ```
///
/// Inputs and outputs are referred to by name. Use `get_inputs` and
/// `get_outputs` to find out which names a model uses, then run it by passing
/// numpy arrays for each input:
///
/// ```python
/// outputs = model.run(None, {"input": input_array})
/// ```
#[pyclass(frozen, module = "rten")]
pub struct Model {
    model: rten::Model,
}

#[pymethods]
impl Model {
    /// Load a model from a file path or the contents of a model file.
    #[new]
    fn new(path_or_bytes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let model = if let Ok(data) = path_or_bytes.cast::<PyBytes>() {
            rten::Model::load(data.as_bytes().to_vec())
        } else {
            let path: PathBuf = path_or_bytes.extract().map_err(|_| {
                PyTypeError::new_err("model must be a file path or the contents of a model file")
            })?;
            rten::Model::load_file(path)
        };

        model
            .map(|model| Model { model })
            .map_err(|err| RtenError::new_err(err.to_string()))
    }

    /// Run the model and return the requested outputs as numpy arrays.
    ///
    /// `output_names` is a list of output names, or `None` to return all of the
    /// model's outputs. `input_feed` maps input names to numpy arrays.
    fn run(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let output_ids = match output_names {
            Some(names) => names
                .iter()
                .map(|name| self.node_id(name))
                .collect::<PyResult<Vec<NodeId>>>()?,
            None => self.model.output_ids().to_vec(),
        };

        let mut input_ids = Vec::with_capacity(input_feed.len());
        let mut input_tensors = Vec::with_capacity(input_feed.len());
        for (name, value) in input_feed.iter() {
            let name: String = name
                .extract()
                .map_err(|_| PyTypeError::new_err("input names must be strings"))?;
            input_ids.push(self.node_id(&name)?);
            input_tensors.push(input_tensor(&name, &value)?);
        }
        let inputs: Vec<(NodeId, ValueOrView)> = input_ids
            .iter()
            .copied()
            .zip(input_tensors.iter().map(|input| input.as_value_or_view()))
            .collect();

        let outputs = py
            .detach(|| self.model.run(inputs, &output_ids, None))
            .map_err(|err| RtenError::new_err(err.to_string()))?;

        outputs
            .into_iter()
            .map(|output| value_to_py(py, output))
            .collect()
    }

    /// Return information about the model's inputs.
    fn get_inputs(&self) -> Vec<NodeInfo> {
        self.node_infos(self.model.input_ids())
    }

    /// Return information about the model's outputs.
    fn get_outputs(&self) -> Vec<NodeInfo> {
        self.node_infos(self.model.output_ids())
    }

    fn __repr__(&self) -> String {
        let names = |nodes: Vec<NodeInfo>| {
            nodes
                .into_iter()
                .map(|node| node.name)
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "Model(inputs=[{}], outputs=[{}])",
            names(self.get_inputs()),
            names(self.get_outputs())
        )
    }
}

impl Model {
    /// Look up a node by name, for use as a model input or output.
    fn node_id(&self, name: &str) -> PyResult<NodeId> {
        self.model
            .find_node(name)
            .ok_or_else(|| PyValueError::new_err(format!("model has no node named \"{}\"", name)))
    }

    fn node_infos(&self, ids: &[NodeId]) -> Vec<NodeInfo> {
        ids.iter()
            .map(|&id| {
                let info = self.model.node_info(id);
                NodeInfo {
                    name: info
                        .as_ref()
                        .and_then(|info| info.name())
                        .unwrap_or_default()
                        .to_string(),
                    dtype: info
                        .as_ref()
                        .and_then(|info| info.dtype())
                        .map(|dtype| dtype.to_string()),
                    shape: info.as_ref().and_then(|info| info.shape()),
                }
            })
            .collect()
    }
}

/// Information about one of a model's inputs or outputs.
#[pyclass(frozen, module = "rten")]
pub struct NodeInfo {
    /// The name used to refer to this input or output in `Model.run`.
    #[pyo3(get)]
    name: String,

    dtype: Option<String>,
    shape: Option<Vec<Dimension>>,
}

#[pymethods]
impl NodeInfo {
    /// The value's type, eg. `"tensor(f32)"`, or `None` if the model doesn't
    /// specify it.
    #[getter]
    fn get_type(&self) -> Option<&str> {
        self.dtype.as_deref()
    }

    /// The value's shape. Dimension sizes can be either fixed values or
    /// symbolic names.
    ///
    /// This is `None` if the model doesn't specify a shape.
    #[getter]
    fn get_shape(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        self.shape
            .as_ref()
            .map(|shape| {
                shape
                    .iter()
                    .map(|dim| match dim {
                        Dimension::Fixed(size) => size.into_py_any(py),
                        Dimension::Symbolic(name) => name.into_py_any(py),
                    })
                    .collect()
            })
            .transpose()
    }

    fn __repr__(&self) -> String {
        let shape = match &self.shape {
            Some(shape) => {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|dim| match dim {
                        Dimension::Fixed(size) => size.to_string(),
                        Dimension::Symbolic(name) => name.clone(),
                    })
                    .collect();
                format!("[{}]", dims.join(", "))
            }
            None => "None".to_string(),
        };
        let dtype = match &self.dtype {
            Some(dtype) => format!("\"{}\"", dtype),
            None => "None".to_string(),
        };
        format!(
            "NodeInfo(name=\"{}\", type={}, shape={})",
            self.name, dtype, shape
        )
    }
}

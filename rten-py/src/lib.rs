//! Python bindings for the [RTen](https://github.com/robertknight/rten)
//! machine learning runtime.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

mod model;
mod value;

create_exception!(
    rten,
    RtenError,
    PyException,
    "Error raised when loading or running a model fails."
);

#[pymodule]
fn _rten(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("RtenError", module.py().get_type::<RtenError>())?;
    module.add_class::<model::Model>()?;
    module.add_class::<model::NodeInfo>()?;
    Ok(())
}

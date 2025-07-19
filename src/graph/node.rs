use std::sync::Arc;

use rten_tensor::prelude::*;
use rten_tensor::{DynLayout, Tensor, TensorView};

use super::NodeId;
use crate::constant_storage::ArcTensorView;
use crate::ops::Operator;
use crate::value::{DataType, ValueView};

#[derive(Debug)]
pub enum Node {
    Operator(OperatorNode),
    Constant(Constant),
    Value(ValueNode),
}

impl Node {
    /// Return the debug name of this node
    pub fn name(&self) -> Option<&str> {
        match self {
            Node::Operator(node) => node.name(),
            Node::Constant(constant) => constant.name(),
            Node::Value(node) => node.name(),
        }
    }

    /// Return the tensor shape associated with this node.
    ///
    /// For constants this is the shape of the tensor. Operator nodes have no
    /// shape. For values (eg. inputs/outputs) this is the expected shape.
    pub fn shape(&self) -> Option<Vec<Dimension>> {
        let dims_from_fixed_shape =
            |shape: &[usize]| shape.iter().copied().map(Dimension::Fixed).collect();

        match self {
            Node::Operator(_) => None,
            Node::Constant(node) => Some(dims_from_fixed_shape(node.layout().shape())),
            Node::Value(node) => node.shape.clone(),
        }
    }

    /// Return the data type associated with this node.
    ///
    /// - For constants this returns the element type of the tensor
    /// - For values this returns the expected element type of the tensor at
    ///   runtime, if known
    /// - For operators this always returns `None`.
    pub fn dtype(&self) -> Option<DataType> {
        match self {
            Node::Value(node) => node.dtype,
            Node::Constant(constant) => Some(constant.dtype()),
            Node::Operator(_) => None,
        }
    }

    /// Return the contained operator, if this an operator node.
    pub fn as_operator(&self) -> Option<&OperatorNode> {
        match self {
            Node::Operator(op) => Some(op),
            _ => None,
        }
    }

    /// Return the contained constant, if this a constant node.
    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Node::Constant(c) => Some(c),
            _ => None,
        }
    }
}

/// Represents the size of a dimension of a runtime-provided value, such as
/// an operator input, output or intermediate value.
#[derive(Clone, Debug, PartialEq)]
pub enum Dimension {
    /// A dimension whose expected size is fixed and specified as part of the
    /// model.
    Fixed(usize),

    /// A dimension whose size is determined at runtime. The symbol provides
    /// a name to identify when different values share a size.
    Symbolic(String),
}

#[derive(Debug)]
pub struct OperatorNode {
    name: Option<String>,
    inputs: Vec<Option<NodeId>>,
    outputs: Vec<Option<NodeId>>,
    operator: Arc<dyn Operator + Send + Sync>,
}

impl OperatorNode {
    pub fn new(
        name: Option<&str>,
        input_ids: &[Option<NodeId>],
        output_ids: &[Option<NodeId>],
        operator: Arc<dyn Operator + Send + Sync>,
    ) -> Self {
        OperatorNode {
            name: name.map(|s| s.to_owned()),
            inputs: Vec::from(input_ids),
            outputs: Vec::from(output_ids),
            operator,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn input_ids(&self) -> &[Option<NodeId>] {
        &self.inputs
    }

    pub fn output_ids(&self) -> &[Option<NodeId>] {
        &self.outputs
    }

    pub fn operator(&self) -> &dyn Operator {
        self.operator.as_ref()
    }

    /// Return a new `Arc` reference to this node's operator.
    ///
    /// Since operators are stateless and immutable once added to a graph, they
    /// can be "cloned" just be creating a new reference.
    pub fn clone_operator(&self) -> Arc<dyn Operator + Send + Sync> {
        self.operator.clone()
    }

    /// Replace an input in the operator's list of inputs.
    ///
    /// Consumers outside the graph module should use graph-level methods instead
    /// which update edge caches.
    pub(super) fn replace_input(&mut self, old_id: NodeId, new_id: NodeId) {
        for input_id in self.inputs.iter_mut() {
            if *input_id == Some(old_id) {
                *input_id = Some(new_id);
            }
        }
    }
}

#[derive(Debug)]
pub struct ValueNode {
    name: Option<String>,
    shape: Option<Vec<Dimension>>,
    dtype: Option<DataType>,
}

impl ValueNode {
    pub fn new(name: Option<&str>, shape: Option<Vec<Dimension>>, dtype: Option<DataType>) -> Self {
        ValueNode {
            name: name.map(|s| s.to_owned()),
            shape,
            dtype,
        }
    }

    /// Return the number of dimensions in this value, if it has shape information.
    pub fn ndim(&self) -> Option<usize> {
        self.shape.as_ref().map(|s| s.len())
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

#[derive(Debug)]
pub enum Constant {
    Float(ConstantNode<f32>),
    Int32(ConstantNode<i32>),
    Int8(ConstantNode<i8>),
    UInt8(ConstantNode<u8>),
}

impl Constant {
    pub fn name(&self) -> Option<&str> {
        match self {
            Constant::Float(f) => f.name.as_deref(),
            Constant::Int32(i) => i.name.as_deref(),
            Constant::Int8(i) => i.name.as_deref(),
            Constant::UInt8(i) => i.name.as_deref(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }

    /// Clone this constant, but only if it can be done so cheaply by
    /// incrementing a ref count on the underlying data.
    pub fn clone_ref(&self) -> Option<Constant> {
        match self {
            Constant::Float(f) => f.clone_ref().map(Constant::Float),
            Constant::Int32(i) => i.clone_ref().map(Constant::Int32),
            Constant::Int8(i) => i.clone_ref().map(Constant::Int8),
            Constant::UInt8(i) => i.clone_ref().map(Constant::UInt8),
        }
    }

    pub fn layout(&self) -> &DynLayout {
        match self {
            Constant::Float(f) => f.layout(),
            Constant::Int32(i) => i.layout(),
            Constant::Int8(i) => i.layout(),
            Constant::UInt8(i) => i.layout(),
        }
    }

    /// Return the data for this constant as a tensor view.
    pub fn as_view(&self) -> ValueView<'_> {
        match self {
            Constant::Float(f) => ValueView::FloatTensor(f.view()),
            Constant::Int32(i) => ValueView::Int32Tensor(i.view()),
            Constant::Int8(i) => ValueView::Int8Tensor(i.view()),
            Constant::UInt8(i) => ValueView::UInt8Tensor(i.view()),
        }
    }

    fn dtype(&self) -> DataType {
        match self {
            Constant::Float(_) => DataType::Float,
            Constant::Int32(_) => DataType::Int32,
            Constant::Int8(_) => DataType::Int8,
            Constant::UInt8(_) => DataType::UInt8,
        }
    }
}

#[derive(Debug)]
pub struct ConstantNode<T> {
    name: Option<String>,
    data: ConstantNodeData<T>,
}

impl<T> ConstantNode<T> {
    pub fn new(name: Option<&str>, data: ConstantNodeData<T>) -> Self {
        ConstantNode {
            name: name.map(|s| s.to_owned()),
            data,
        }
    }

    pub fn view(&self) -> TensorView<'_, T> {
        match &self.data {
            ConstantNodeData::Owned(data) => data.view(),
            ConstantNodeData::Arc(data) => data.view(),
        }
    }

    fn clone_ref(&self) -> Option<ConstantNode<T>> {
        let data = self.data.clone_ref()?;
        Some(ConstantNode {
            name: self.name.clone(),
            data,
        })
    }

    fn layout(&self) -> &DynLayout {
        match &self.data {
            ConstantNodeData::Owned(data) => data.layout(),
            ConstantNodeData::Arc(data) => data.layout(),
        }
    }
}

macro_rules! impl_constant_node {
    ($scalar_type:ty, $variant:ident) => {
        impl From<ConstantNode<$scalar_type>> for Constant {
            fn from(node: ConstantNode<$scalar_type>) -> Constant {
                Constant::$variant(node)
            }
        }
    };
}

impl_constant_node!(f32, Float);
impl_constant_node!(i32, Int32);
impl_constant_node!(i8, Int8);
impl_constant_node!(u8, UInt8);

/// Data for a constant node (ie. model weights) in a [`Graph`].
#[derive(Debug)]
pub enum ConstantNodeData<T> {
    Owned(Tensor<T>),
    Arc(ArcTensorView<T>),
}

impl<T> ConstantNodeData<T> {
    fn clone_ref(&self) -> Option<ConstantNodeData<T>> {
        match self {
            ConstantNodeData::Owned(_) => None,
            ConstantNodeData::Arc(view) => Some(ConstantNodeData::Arc(view.clone())),
        }
    }
}

impl<T> From<Tensor<T>> for ConstantNodeData<T> {
    fn from(val: Tensor<T>) -> ConstantNodeData<T> {
        ConstantNodeData::Owned(val)
    }
}

impl<T> From<ArcTensorView<T>> for ConstantNodeData<T> {
    fn from(val: ArcTensorView<T>) -> ConstantNodeData<T> {
        ConstantNodeData::Arc(val)
    }
}

/// Extract typed data from a [`Constant`].
pub trait TypedConstant<T> {
    fn as_view(&self) -> Option<TensorView<'_, T>>;
    fn as_scalar(&self) -> Option<T>;
    fn as_vector(&self) -> Option<&[T]>;
}

macro_rules! impl_typed_constant {
    ($type:ty, $variant:ident) => {
        impl TypedConstant<$type> for Constant {
            fn as_view(&self) -> Option<TensorView<'_, $type>> {
                match self {
                    Constant::$variant(tensor) => Some(tensor.view()),
                    _ => None,
                }
            }

            fn as_scalar(&self) -> Option<$type> {
                TypedConstant::as_view(self).and_then(|view| view.item().copied())
            }

            fn as_vector(&self) -> Option<&[$type]> {
                TypedConstant::as_view(self).and_then(|view| match (view.ndim(), view.data()) {
                    (1, Some(vec_data)) => Some(vec_data),
                    _ => None,
                })
            }
        }
    };
}

impl_typed_constant!(f32, Float);
impl_typed_constant!(i32, Int32);
impl_typed_constant!(i8, Int8);
impl_typed_constant!(u8, UInt8);

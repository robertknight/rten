use std::fmt;

use serde::de::{Deserialize, Deserializer, Error, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use crate::iterators::Iter;
use crate::{AsView, Layout, MutLayout, Storage, TensorBase};

struct TensorData<'a, T> {
    iter: Iter<'a, T>,
}

impl<T> Serialize for TensorData<'_, T>
where
    T: Serialize,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        serializer.collect_seq(self.iter.clone())
    }
}

impl<S: Storage, L: MutLayout> Serialize for TensorBase<S, L>
where
    S::Elem: Serialize,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        let mut tensor = serializer.serialize_struct("Tensor", 2)?;
        tensor.serialize_field("shape", self.shape().as_ref())?;
        tensor.serialize_field("data", &TensorData { iter: self.iter() })?;
        tensor.end()
    }
}

struct TensorVisitor<T, L> {
    data_marker: std::marker::PhantomData<T>,
    layout_marker: std::marker::PhantomData<L>,
}

impl<'de, T, L> Visitor<'de> for TensorVisitor<T, L>
where
    T: Deserialize<'de>,
    L: MutLayout,
    for<'a> L::Index<'a>: TryFrom<&'a [usize]>,
{
    type Value = TensorBase<Vec<T>, L>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "a tensor with \"shape\" and \"data\" fields")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut data: Option<Vec<T>> = None;
        let mut shape: Option<Vec<usize>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "data" => {
                    if data.is_some() {
                        return Err(A::Error::duplicate_field("data"));
                    }
                    data = Some(map.next_value()?);
                }
                "shape" => {
                    if shape.is_some() {
                        return Err(A::Error::duplicate_field("shape"));
                    }
                    shape = Some(map.next_value()?);
                }
                _ => {
                    return Err(A::Error::unknown_field(&key, &["data", "shape"]));
                }
            }
        }

        let Some(shape) = shape else {
            return Err(A::Error::missing_field("shape"));
        };
        let Some(data) = data else {
            return Err(A::Error::missing_field("data"));
        };

        let Ok(shape_ref): Result<L::Index<'_>, _> = shape.as_slice().try_into() else {
            return Err(A::Error::custom("incorrect shape length for tensor rank"));
        };

        TensorBase::try_from_data(shape_ref, data)
            .map_err(|_| A::Error::custom("data length does not match shape product"))
    }
}

impl<'de, T, L: MutLayout> Deserialize<'de> for TensorBase<Vec<T>, L>
where
    T: Deserialize<'de>,
    for<'a> L::Index<'a>: TryFrom<&'a [usize]>,
{
    fn deserialize<D>(deserializer: D) -> Result<TensorBase<Vec<T>, L>, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tensor",
            &["shape", "data"],
            TensorVisitor::<T, L> {
                data_marker: std::marker::PhantomData,
                layout_marker: std::marker::PhantomData,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{NdTensor, Tensor};

    #[test]
    fn test_deserialize_serialize_dynamic_rank() {
        struct Case<'a> {
            json: &'a str,
            expected: Result<Tensor<f32>, String>,
        }

        let cases = [
            Case {
                json: "[]",
                expected: Err(format!(
                    "expected a tensor with \"shape\" and \"data\" fields"
                )),
            },
            Case {
                json: r#"{"data":[]}"#,
                expected: Err(format!("missing field `shape`")),
            },
            Case {
                json: r#"{"data":[], "data": []}"#,
                expected: Err(format!("duplicate field `data`")),
            },
            Case {
                json: r#"{"shape":[]}"#,
                expected: Err(format!("missing field `data`")),
            },
            Case {
                json: r#"{"shape":[], "shape": []}"#,
                expected: Err(format!("duplicate field `shape`")),
            },
            Case {
                json: r#"{"data": [1.0, 0.5, 2.0, 1.5], "shape": [2, 2]}"#,
                expected: Ok(Tensor::from([[1.0, 0.5], [2.0, 1.5]])),
            },
            Case {
                json: r#"{"data": [1.0, 0.5, 2.0, 1.5], "shape": [2, 3]}"#,
                expected: Err(format!("data length does not match shape product")),
            },
        ];

        for Case { json, expected } in cases {
            let actual: Result<Tensor<f32>, String> =
                serde_json::from_str(&json).map_err(|e| e.to_string());
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual, expected);

                    // Verify that serializing the result produces the original
                    // JSON.
                    let actual_json = serde_json::to_value(actual).unwrap();
                    let expected_json: serde_json::Value = serde_json::from_str(&json).unwrap();
                    assert_eq!(actual_json, expected_json);
                }
                (Err(actual_err), Err(expected_err)) => assert!(
                    actual_err.contains(&expected_err),
                    "expected \"{}\" to contain \"{}\"",
                    actual_err,
                    expected_err
                ),
                (actual, expected) => assert_eq!(actual, expected),
            }
        }
    }

    #[test]
    fn test_deserialize_serialize_static_rank() {
        struct Case<'a> {
            json: &'a str,
            expected: Result<NdTensor<f32, 2>, String>,
        }

        let cases = [
            Case {
                json: r#"{"data": [1.0, 0.5, 2.0, 1.5], "shape": [2, 2]}"#,
                expected: Ok(NdTensor::from([[1.0, 0.5], [2.0, 1.5]])),
            },
            Case {
                json: r#"{"data": [1.0, 0.5, 2.0, 1.5], "shape": [1, 2, 2]}"#,
                expected: Err(format!("incorrect shape length for tensor rank")),
            },
        ];

        for Case { json, expected } in cases {
            let actual: Result<NdTensor<f32, 2>, String> =
                serde_json::from_str(&json).map_err(|e| e.to_string());

            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual, expected);

                    // Verify that serializing the result produces the original
                    // JSON.
                    let actual_json = serde_json::to_value(actual).unwrap();
                    let expected_json: serde_json::Value = serde_json::from_str(&json).unwrap();
                    assert_eq!(actual_json, expected_json);
                }
                (Err(actual_err), Err(expected_err)) => assert!(
                    actual_err.contains(&expected_err),
                    "expected \"{}\" to contain \"{}\"",
                    actual_err,
                    expected_err
                ),
                (actual, expected) => assert_eq!(actual, expected),
            }
        }
    }
}

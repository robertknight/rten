use std::collections::HashMap;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MetadataField {
    // Standard fields in ONNX models.
    ProducerName,
    ProducerVersion,

    Custom(String),
}

impl MetadataField {
    fn name(&self) -> &str {
        match self {
            Self::ProducerName => "producer_name",
            Self::ProducerVersion => "producer_version",
            Self::Custom(value) => value,
        }
    }
}

impl std::fmt::Display for MetadataField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Collection of (name, value) metadata entries for a model.
///
/// This provides access to
/// [metadata](https://onnx.ai/onnx/intro/concepts.html#metadata) recorded in an
/// ONNX model such as the version, license and tools used to produce the model.
///
/// There are methods for standard fields. Other fields can be looked up by
/// name using [`get`](Self::get). The [`fields`](Self::fields) method returns
/// an iterator over all fields.
#[derive(Debug, Default)]
pub struct ModelMetadata {
    fields: HashMap<MetadataField, String>,
}

impl ModelMetadata {
    pub(crate) fn from_fields(fields: impl IntoIterator<Item = (MetadataField, String)>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// Return the name of the framework or tool used to produce the model.
    pub fn producer_name(&self) -> Option<&str> {
        self.field(&MetadataField::ProducerName)
    }

    /// Return the version of the framework or tool used to produce the model.
    pub fn producer_version(&self) -> Option<&str> {
        self.field(&MetadataField::ProducerVersion)
    }

    /// Get the value of a custom metadata field by name.
    pub fn get(&self, name: &str) -> Option<&str> {
        let key = MetadataField::Custom(name.to_string());
        self.field(&key)
    }

    fn field(&self, field: &MetadataField) -> Option<&str> {
        self.fields.get(field).map(|x| x.as_str())
    }

    /// Return an iterator over (name, value) pairs for metadata fields.
    ///
    /// The order of fields is not guaranteed.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields
            .iter()
            .map(|(field, val)| (field.name(), val.as_str()))
    }
}

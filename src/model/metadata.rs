use std::collections::HashMap;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MetadataField {
    // Standard fields in rten models.
    CodeRepository,
    Commit,
    Description,
    License,
    ModelRepository,
    OnnxHash,
    RunId,
    RunUrl,

    // Standard fields in ONNX models.
    ProducerName,
    ProducerVersion,
}

impl MetadataField {
    fn name(&self) -> &str {
        match self {
            Self::CodeRepository => "code_repository",
            Self::Commit => "commit",
            Self::Description => "description",
            Self::License => "license",
            Self::ModelRepository => "model_repository",
            Self::OnnxHash => "onnx_hash",
            Self::RunId => "run_id",
            Self::RunUrl => "run_url",
            Self::ProducerName => "producer_name",
            Self::ProducerVersion => "producer_version",
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
/// The available metadata depends on the model format and tool used to create
/// the model. RTen format models have a standard set of fields. For ONNX models
/// the metadata is more free-form.
///
/// There are methods for standard fields. The [`fields`](Self::fields) method
/// returns an iterator over all fields.
#[derive(Default)]
pub struct ModelMetadata {
    fields: HashMap<MetadataField, String>,
}

impl ModelMetadata {
    pub(crate) fn from_fields(fields: impl IntoIterator<Item = (MetadataField, String)>) -> Self {
        Self {
            fields: fields.into_iter().collect(),
        }
    }

    /// Return the SHA-256 hash of the ONNX model used to generate this RTen
    /// model.
    ///
    /// This is used for .rten format models only.
    pub fn onnx_hash(&self) -> Option<&str> {
        self.field(&MetadataField::OnnxHash)
    }

    /// Return a short description of what this model does.
    ///
    /// This is used for .rten format models only.
    pub fn description(&self) -> Option<&str> {
        self.field(&MetadataField::Description)
    }

    /// Return the license identifier for this model. It is recommended that
    /// this be an SPDX identifier.
    ///
    /// This is used for .rten format models only.
    pub fn license(&self) -> Option<&str> {
        self.field(&MetadataField::License)
    }

    /// Return the commit from the repository referenced by
    /// [code_repository](ModelMetadata::code_repository) which was used to
    /// create this model.
    ///
    /// This is used for .rten format models only.
    pub fn commit(&self) -> Option<&str> {
        self.field(&MetadataField::Commit)
    }

    /// Return the URL of the repository (eg. on GitHub) containing the model's
    /// code.
    ///
    /// This is used for .rten format models only.
    pub fn code_repository(&self) -> Option<&str> {
        self.field(&MetadataField::CodeRepository)
    }

    /// Return the URL of the repository (eg. on Hugging Face) where the model
    /// is hosted.
    ///
    /// This is used for .rten format models only.
    pub fn model_repository(&self) -> Option<&str> {
        self.field(&MetadataField::ModelRepository)
    }

    /// Return the ID of the training run that produced this model.
    ///
    /// When models are developed using experiment tracking services such as
    /// Weights and Biases, this enables looking up the training run that
    /// produced the model.
    ///
    /// This is used for .rten format models only.
    pub fn run_id(&self) -> Option<&str> {
        self.field(&MetadataField::RunId)
    }

    /// Return a URL for the training run that produced this model.
    ///
    /// When models are developed using experiment tracking services such as
    /// Weights and Biases, this enables looking up the training run that
    /// produced the model.
    ///
    /// This is used for .rten format models only.
    pub fn run_url(&self) -> Option<&str> {
        self.field(&MetadataField::RunUrl)
    }

    /// Return the name of the framework or tool used to produce the model.
    pub fn producer_name(&self) -> Option<&str> {
        self.field(&MetadataField::ProducerName)
    }

    /// Return the version of the framework or tool used to produce the model.
    pub fn producer_version(&self) -> Option<&str> {
        self.field(&MetadataField::ProducerVersion)
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

#[cfg(test)]
mod tests {
    use super::{MetadataField, ModelMetadata};

    #[test]
    fn test_model_metadata() {
        let model_metadata = ModelMetadata::from_fields([
            (MetadataField::OnnxHash, "abc".to_string()),
            (MetadataField::Description, "A simple model".to_string()),
            (MetadataField::License, "BSD-2-Clause".to_string()),
            (MetadataField::Commit, "def".to_string()),
            (
                MetadataField::CodeRepository,
                "https://github.com/robertknight/rten".to_string(),
            ),
            (
                MetadataField::ModelRepository,
                "https://huggingface.co/robertknight/rten".to_string(),
            ),
            (MetadataField::RunId, "1234".to_string()),
            (
                MetadataField::RunUrl,
                "https://wandb.ai/robertknight/text-detection/runs/1234".to_string(),
            ),
        ]);

        assert_eq!(model_metadata.onnx_hash(), Some("abc"));
        assert_eq!(model_metadata.description(), Some("A simple model"));
        assert_eq!(model_metadata.license(), Some("BSD-2-Clause"));
        assert_eq!(model_metadata.commit(), Some("def"));
        assert_eq!(
            model_metadata.code_repository(),
            Some("https://github.com/robertknight/rten")
        );
        assert_eq!(
            model_metadata.model_repository(),
            Some("https://huggingface.co/robertknight/rten")
        );
        assert_eq!(model_metadata.run_id(), Some("1234"));
        assert_eq!(
            model_metadata.run_url(),
            Some("https://wandb.ai/robertknight/text-detection/runs/1234")
        );
    }
}

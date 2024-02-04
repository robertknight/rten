use crate::schema_generated as sg;

/// Metadata for an RTen model.
///
/// This provides access to information such as:
///
///  - The ONNX model that was used to generate it
///  - The license
///  - Details of the training run that produced the model
///  - Related URLs
#[derive(Default)]
pub struct ModelMetadata {
    onnx_hash: Option<String>,
    description: Option<String>,
    license: Option<String>,
    commit: Option<String>,
    code_repository: Option<String>,
    model_repository: Option<String>,
    run_id: Option<String>,
    run_url: Option<String>,
}

impl ModelMetadata {
    /// Deserialize a ModelMetadata from data in a flatbuffers file.
    pub(crate) fn deserialize(metadata: sg::Metadata<'_>) -> ModelMetadata {
        ModelMetadata {
            onnx_hash: metadata.onnx_hash().map(|s| s.to_string()),
            description: metadata.description().map(|s| s.to_string()),
            license: metadata.license().map(|s| s.to_string()),
            commit: metadata.commit().map(|s| s.to_string()),
            code_repository: metadata.code_repository().map(|s| s.to_string()),
            model_repository: metadata.model_repository().map(|s| s.to_string()),
            run_id: metadata.run_id().map(|s| s.to_string()),
            run_url: metadata.run_url().map(|s| s.to_string()),
        }
    }

    /// Return the SHA-256 hash of the ONNX model used to generate this RTen
    /// model.
    pub fn onnx_hash(&self) -> Option<&str> {
        self.onnx_hash.as_deref()
    }

    /// Return a short description of what this model does.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Return the license identifier for this model. It is recommended that
    /// this be an SPDX identifier.
    pub fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    /// Return the commit from the repository referenced by
    /// [code_repository](ModelMetadata::code_repository) which was used to
    /// create this model.
    pub fn commit(&self) -> Option<&str> {
        self.commit.as_deref()
    }

    /// Return the URL of the repository (eg. on GitHub) containing the model's
    /// code.
    pub fn code_repository(&self) -> Option<&str> {
        self.code_repository.as_deref()
    }

    /// Return the URL of the repository (eg. on Hugging Face) where the model
    /// is hosted.
    pub fn model_repository(&self) -> Option<&str> {
        self.model_repository.as_deref()
    }

    /// Return the ID of the training run that produced this model.
    ///
    /// When models are developed using experiment tracking services such as
    /// Weights and Biases, this enables looking up the training run that
    /// produced the model.
    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    /// Return a URL for the training run that produced this model.
    ///
    /// When models are developed using experiment tracking services such as
    /// Weights and Biases, this enables looking up the training run that
    /// produced the model.
    pub fn run_url(&self) -> Option<&str> {
        self.run_url.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::ModelMetadata;
    use crate::schema_generated as sg;
    use flatbuffers::FlatBufferBuilder;

    #[test]
    fn test_model_metadata() {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let onnx_hash = builder.create_string("abc");
        let description = builder.create_string("A simple model");
        let license = builder.create_string("BSD-2-Clause");
        let commit = builder.create_string("def");
        let code_repository = builder.create_string("https://github.com/robertknight/rten");
        let model_repository = builder.create_string("https://huggingface.co/robertknight/rten");
        let run_id = builder.create_string("1234");
        let run_url =
            builder.create_string("https://wandb.ai/robertknight/text-detection/runs/1234");

        let mut meta_builder = sg::MetadataBuilder::new(&mut builder);
        meta_builder.add_onnx_hash(onnx_hash);
        meta_builder.add_description(description);
        meta_builder.add_license(license);
        meta_builder.add_commit(commit);
        meta_builder.add_code_repository(code_repository);
        meta_builder.add_model_repository(model_repository);
        meta_builder.add_run_id(run_id);
        meta_builder.add_run_url(run_url);
        let metadata = meta_builder.finish();

        builder.finish_minimal(metadata);
        let data = builder.finished_data();

        let deserialized_meta = flatbuffers::root::<sg::Metadata>(&data).unwrap();
        let model_metadata = ModelMetadata::deserialize(deserialized_meta);

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

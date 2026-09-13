"""Model metadata shared by the ONNX and rten model formats."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Metadata:
    """
    Model metadata.

    This corresponds to the `ModelMetadata` struct in RTen. See its docs for
    details of the individual fields.

    When adding new fields here, they also need to be added to
    `METADATA_BUILDER_FNS` in `rten_convert.converter`.
    """

    code_repository: Optional[str] = None
    commit: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    model_repository: Optional[str] = None
    onnx_hash: Optional[str] = None
    run_id: Optional[str] = None
    run_url: Optional[str] = None

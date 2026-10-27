"""File management and validation service for Business Entity Resolution API."""

import io
import shutil
import uuid
from pathlib import Path
from typing import Tuple
import pandas as pd
from fastapi import UploadFile


class FileValidationError(Exception):
    """Exception raised when an uploaded file fails format or schema validation."""

    def __init__(self, error: str, message: str):
        super().__init__(message)
        self.error = error
        self.message = message


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def get_project_root() -> Path:
    """
    Locates the project root directory containing 'models', 'output', and 'dataset'.
    """
    script_dir = Path(__file__).resolve().parent
    # Check parent dirs
    for candidate in [script_dir.parent.parent, script_dir.parent, Path.cwd()]:
        if (candidate / "models").exists() and (candidate / "output").exists():
            return candidate

    nested = Path.cwd() / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
    if (nested / "models").exists():
        return nested

    nested_single = Path.cwd() / "synthetic_entity_resolution_dataset"
    if (nested_single / "models").exists():
        return nested_single

    return script_dir.parent.parent


def get_temp_upload_dir() -> Path:
    """Returns directory for storing uploaded files temporarily."""
    temp_dir = get_project_root() / "backend" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_download_path(filename: str) -> Path:
    """Returns absolute path to downloadable output file."""
    root = get_project_root()
    output_dir = root / "output"
    return output_dir / filename


async def validate_and_load_tsv(
    upload_file: UploadFile,
    source_label: str
) -> Tuple[pd.DataFrame, Path]:
    """
    Validates uploaded file against requirements:
    - Must be provided
    - Filename must end with .tsv
    - Content must be readable TSV
    - Must contain required columns: entity_id, business_name, business_address, country
    - Must contain at least one record

    Saves uploaded file to a controlled temporary directory.
    Returns: (df, saved_temp_path)
    """
    if not upload_file or not upload_file.filename:
        raise FileValidationError(
            error=f"Missing {source_label} file",
            message=f"No file was provided for {source_label}. Please upload a .tsv file."
        )

    filename_lower = upload_file.filename.lower()
    if not filename_lower.endswith(".tsv"):
        raise FileValidationError(
            error=f"Invalid {source_label} file",
            message=f"Uploaded file '{upload_file.filename}' must have a .tsv extension."
        )

    try:
        content_bytes = await upload_file.read()
    except Exception as e:
        raise FileValidationError(
            error=f"Read error in {source_label}",
            message=f"Failed to read file contents: {str(e)}"
        )

    if not content_bytes or len(content_bytes.strip()) == 0:
        raise FileValidationError(
            error=f"Empty {source_label} file",
            message=f"The uploaded file '{upload_file.filename}' is empty."
        )

    try:
        df = pd.read_csv(
            io.BytesIO(content_bytes),
            sep="\t",
            dtype=str,
            keep_default_na=False
        )
    except Exception as e:
        raise FileValidationError(
            error=f"TSV parsing error in {source_label}",
            message=f"Could not parse '{upload_file.filename}' as a tab-separated TSV: {str(e)}"
        )

    # Validate required columns
    present_cols = set(df.columns)
    for col in REQUIRED_COLUMNS:
        if col not in present_cols:
            raise FileValidationError(
                error=f"Invalid {source_label} file",
                message=f"Missing required column: {col}"
            )

    # Validate non-empty records
    if len(df) == 0:
        raise FileValidationError(
            error=f"Empty dataset in {source_label}",
            message=f"The uploaded file '{upload_file.filename}' has 0 rows of data."
        )

    # Store uploaded file temporarily in a controlled directory
    temp_dir = get_temp_upload_dir()
    session_id = uuid.uuid4().hex[:8]
    safe_stem = Path(upload_file.filename).stem[:20]
    safe_filename = f"{source_label}_{session_id}_{safe_stem}.tsv"
    saved_path = temp_dir / safe_filename

    with open(saved_path, "wb") as f:
        f.write(content_bytes)

    return df, saved_path

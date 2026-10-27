"""Response models and schemas for Business Entity Resolution API."""

from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Service status")
    service: str = Field(
        default="Business Entity Resolution API",
        description="Service name"
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model: str = Field(default="XGBoost", description="Model architecture")
    task: str = Field(default="Binary Entity Classification", description="Task type")
    classes: str = Field(default="Different Business / Same Business", description="Target classification classes")
    threshold: float = Field(description="Optimal decision threshold")
    feature_count: int = Field(description="Number of engineered features used by model")
    status: str = Field(description="Current model loading status")
    metric: Optional[str] = Field(default="macro_entity_f0_5", description="Optimization metric")
    features: Optional[List[str]] = Field(default=None, description="List of feature names")


class ProcessResponse(BaseModel):
    """Pipeline execution completion response."""
    success: bool = Field(default=True, description="Whether processing succeeded")
    message: str = Field(default="Entity matching completed successfully", description="Status message")
    status: str = Field(default="completed", description="Job status")
    source1_records: int = Field(description="Total valid records in Source 1")
    source2_records: int = Field(description="Total valid records in Source 2")
    source3_records: int = Field(description="Total valid records in Source 3")
    candidate_pairs: int = Field(description="Total candidate pairs generated after blocking")
    matched_entities: int = Field(description="Source 1 entities with at least one match")
    unmatched_entities: int = Field(description="Source 1 entities with zero matches (singletons)")
    threshold: float = Field(description="Applied classification threshold")
    statistics: Optional[dict] = Field(default=None, description="Detailed execution statistics summary")


class ResultsResponse(BaseModel):
    """Most recent job execution results summary."""
    success: bool = Field(default=True, description="Success flag")
    status: str = Field(description="Status of most recent job (e.g. idle, completed, failed)")
    message: Optional[str] = Field(default=None, description="Optional informational message")
    source1_count: int = Field(default=0, description="Source 1 record count")
    source2_count: int = Field(default=0, description="Source 2 record count")
    source3_count: int = Field(default=0, description="Source 3 record count")
    candidate_pair_count: int = Field(default=0, description="Candidate pair count")
    matched_entity_count: int = Field(default=0, description="Matched Source 1 entity count")
    unmatched_entity_count: int = Field(default=0, description="Unmatched Source 1 entity count")
    threshold: Optional[float] = Field(default=None, description="Applied threshold")
    matching_file_available: bool = Field(default=False, description="Whether matching_results.tsv is ready for download")
    candidates_file_available: bool = Field(default=False, description="Whether candidate_pairs.tsv is ready for download")
    statistics: Optional[dict] = Field(default=None, description="Summary statistics")
    results: Optional[List[dict]] = Field(default=None, description="Detailed matching results")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error category or summary")
    message: str = Field(description="Detailed error explanation")


class PreviewRow(BaseModel):
    """Single row in matching results preview."""
    source1_entity_id: str = Field(description="Source 1 Entity ID")
    matched_entity_ids: str = Field(description="Comma-separated matched candidate IDs or empty string")
    count: int = Field(description="Number of matched candidate IDs")


class PreviewResponse(BaseModel):
    """Matching results preview response."""
    status: str = Field(description="Preview data status")
    total_rows: int = Field(description="Total rows in matching results")
    preview_limit: int = Field(description="Number of preview rows returned")
    rows: List[PreviewRow] = Field(default_factory=list, description="List of preview matching rows")


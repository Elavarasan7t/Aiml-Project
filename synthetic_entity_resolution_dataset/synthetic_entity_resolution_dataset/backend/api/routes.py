"""API routes for Business Entity Resolution backend."""

from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse

from backend.schemas.response_models import (
    HealthResponse,
    ModelInfoResponse,
    ProcessResponse,
    ResultsResponse,
    ErrorResponse,
    PreviewResponse,
)
from backend.services.file_service import (
    FileValidationError,
    validate_and_load_tsv,
    get_download_path,
)
from backend.services.pipeline_service import (
    pipeline_service,
    PipelineExecutionError,
)

router = APIRouter(prefix="/api", tags=["Entity Resolution"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns health status of the Business Entity Resolution API service."
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="Business Entity Resolution API"
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get ML Model Metadata",
    description="Returns loaded model information, feature count, and selected validation threshold."
)
async def get_model_info():
    """Retrieves model and threshold metadata."""
    try:
        info = pipeline_service.get_model_info()
        return ModelInfoResponse(**info)
    except PipelineExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": e.error, "message": e.message}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Model Error", "message": f"Unable to read model metadata: {str(e)}"}
        )


@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error in uploaded files"},
        500: {"model": ErrorResponse, "description": "Pipeline execution error"}
    },
    summary="Process Uploaded Source Datasets",
    description="Upload Source 1, Source 2, and Source 3 TSV files to execute the full Entity Resolution pipeline."
)
async def process_datasets(
    source1: UploadFile = File(..., description="Source 1 TSV file (e.g. source1.tsv)"),
    source2: UploadFile = File(..., description="Source 2 TSV file (e.g. source2.tsv)"),
    source3: UploadFile = File(..., description="Source 3 TSV file (e.g. source3.tsv)"),
):
    """
    Accepts three TSV files, validates headers and format, executes cleaning, blocking,
    feature engineering, and XGBoost matching, and outputs results.
    """
    # 1. Validate and load each uploaded file
    try:
        df_s1, _ = await validate_and_load_tsv(source1, "Source 1")
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error, "message": e.message}
        )

    try:
        df_s2, _ = await validate_and_load_tsv(source2, "Source 2")
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error, "message": e.message}
        )

    try:
        df_s3, _ = await validate_and_load_tsv(source3, "Source 3")
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error, "message": e.message}
        )

    # 2. Run the Entity Resolution pipeline
    try:
        result = pipeline_service.run_pipeline(df_s1, df_s2, df_s3)
        return ProcessResponse(**result)
    except PipelineExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": e.error, "message": e.message}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Pipeline Processing Error",
                "message": f"An unexpected error occurred during execution: {str(e)}"
            }
        )


@router.get(
    "/results",
    response_model=ResultsResponse,
    summary="Get Latest Processing Job Results",
    description="Returns the status and summary metrics of the most recent Entity Resolution run."
)
async def get_results():
    """Retrieves in-memory job status of most recent processing execution."""
    results = pipeline_service.get_latest_results()
    return ResultsResponse(**results)


@router.get(
    "/results/preview",
    response_model=PreviewResponse,
    summary="Get Matching Results Preview",
    description="Returns top matching results rows for preview display in frontend table."
)
async def get_results_preview(limit: int = 50):
    """Retrieves top N matching results rows."""
    preview_data = pipeline_service.get_matching_preview(limit=limit)
    return PreviewResponse(**preview_data)


@router.get(
    "/download/matching",
    summary="Download Matching Results TSV",
    description="Downloads the generated matching_results.tsv file."
)
async def download_matching_results():
    """Downloads matching_results.tsv."""
    file_path = get_download_path("matching_results.tsv")
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "File Not Found",
                "message": "matching_results.tsv has not been generated yet. Please submit datasets to /api/process first."
            }
        )

    return FileResponse(
        path=str(file_path),
        filename="matching_results.tsv",
        media_type="text/tab-separated-values"
    )


@router.get(
    "/download/candidates",
    summary="Download Candidate Pairs TSV",
    description="Downloads the generated candidate_pairs.tsv file."
)
async def download_candidate_results():
    """Downloads candidate_pairs.tsv."""
    file_path = get_download_path("candidate_pairs.tsv")
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "File Not Found",
                "message": "candidate_pairs.tsv has not been generated yet. Please submit datasets to /api/process first."
            }
        )

    return FileResponse(
        path=str(file_path),
        filename="candidate_pairs.tsv",
        media_type="text/tab-separated-values"
    )

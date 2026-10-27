"""FastAPI Main Application for Business Entity Resolution."""

import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Ensure project root is in sys.path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
for p in [str(parent_dir), str(script_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.api.routes import router as api_router
from backend.services.file_service import FileValidationError
from backend.services.pipeline_service import PipelineExecutionError

app = FastAPI(
    title="Business Entity Resolution API",
    description=(
        "REST API for multi-source Business Entity Resolution. "
        "Provides endpoints for ML model inspection, file processing, and results download."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React / frontend development
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FileValidationError)
async def file_validation_handler(request: Request, exc: FileValidationError):
    """Handles uploaded file validation errors with clean HTTP 400 responses."""
    return JSONResponse(
        status_code=400,
        content={"error": exc.error, "message": exc.message},
    )


@app.exception_handler(PipelineExecutionError)
async def pipeline_error_handler(request: Request, exc: PipelineExecutionError):
    """Handles ML pipeline errors with HTTP 500 without leaking stack traces."""
    return JSONResponse(
        status_code=500,
        content={"error": exc.error, "message": exc.message},
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Ensures consistent dictionary error payload for HTTP exceptions."""
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP Error", "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors safely without leaking internal server stack traces."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request.",
        },
    )


# Include API routes
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from pydantic import BaseModel, Field
from typing import Optional
from .model import text_generator

# Define request/response models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_length: Optional[int] = Field(default=100, ge=10, le=2000)

class TextGenerationResponse(BaseModel):
    generated_text: str

# Initialize FastAPI app
builder = FastAPI(
    title="Text Generation API",
    description="API for generating text using language models",
    version="1.0.0"
)

# Add CORS middleware
builder.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Custom middleware for error handling
@builder.middleware("http")
async def custom_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "detail": str(e)}
        )

# Health check endpoint
@builder.get("/")
async def read_root():
    if not builder:
        raise HTTPException(status_code=500, detail="Server is not initialized")
    return {"status": "healthy", "service": "text-generation"}

# Text generation endpoint
@builder.post("/generate-text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    if not text_generator:
        raise HTTPException(
            status_code=500, 
            detail="Text generator is not initialized"
        )
    
    try:
        generated_text = text_generator.generate_text(
            request.prompt, 
            request.max_length
        )
        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

# Error handlers
@builder.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@builder.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )
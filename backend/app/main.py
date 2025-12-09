# main.py - FastAPI application for TaskMind multi-modal classification

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add multimodal directory to path
sys.path.insert(0, str(Path(__file__).parent / "multimodal"))

from pipeline import classify_video_pipeline

app = FastAPI(
    title="TaskMind Multi-Modal Video Classifier API",
    description="Classify YouTube videos using Computer Vision, Audio Transcription, and Metadata",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoClassificationRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL")
    title: str = Field(..., description="Video title")
    description: str = Field(default="", description="Video description")
    tags: str = Field(default="", description="Video tags (comma-separated)")
    num_frames: int = Field(default=6, description="Number of frames to extract for CV analysis")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional weights for each model: {'cv': 0.33, 'audio': 0.33, 'metadata': 0.34}"
    )


class ClassificationResponse(BaseModel):
    cv_classification: Dict[str, float] = Field(..., description="Computer Vision classification scores")
    audio_classification: Dict[str, float] = Field(..., description="Audio transcription classification scores")
    metadata_classification: Dict[str, float] = Field(..., description="Metadata classification scores")
    final_classification: Dict[str, float] = Field(..., description="Final aggregated classification scores")
    top_category: str = Field(..., description="Category with highest final score")
    top_confidence: float = Field(..., description="Confidence score for top category")
    cv_top_5: List[tuple] = Field(..., description="Top 5 categories from CV model")
    audio_top_5: List[tuple] = Field(..., description="Top 5 categories from Audio model")
    metadata_top_5: List[tuple] = Field(..., description="Top 5 categories from Metadata model")
    final_top_5: List[tuple] = Field(..., description="Top 5 categories from final aggregation")


@app.get("/")
async def root():
    return {
        "message": "TaskMind Multi-Modal Video Classifier API",
        "version": "2.0.0",
        "endpoints": {
            "/classify": "POST - Classify a video using all three modalities",
            "/docs": "GET - API documentation"
        }
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_video(request: VideoClassificationRequest):
    """
    Classify a YouTube video using three modalities:
    1. Computer Vision (CV) - Analyzes 6 extracted frames
    2. Audio - Transcribes audio and classifies transcription
    3. Metadata - Classifies title, description, and tags
    
    Returns classification scores from each modality and a final aggregated result.
    """
    try:
        result = classify_video_pipeline(
            video_url=request.video_url,
            title=request.title,
            description=request.description,
            tags=request.tags,
            num_frames=request.num_frames,
            weights=request.weights
        )
        
        return ClassificationResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TaskMind API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



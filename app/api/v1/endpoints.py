import logging
from typing import Dict
from fastapi import APIRouter, HTTPException, status
from app.api.v1.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.modalities.face import FacialAnalyzer
from app.modalities.fusion import ModalityFusionEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Modality Registry
MODALITY_ANALYZERS = {
    "face": FacialAnalyzer()
    # Extension point: "voice": VoiceAnalyzer() can be registered here in the future
}

fusion_engine = ModalityFusionEngine()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
@router.get("/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint exposing service state and supported modalities."""
    return HealthResponse(
        status="healthy",
        service="student-mental-health-facial-analysis",
        version="1.0.0",
        supported_modalities=list(MODALITY_ANALYZERS.keys())
    )


@router.post(
    "/v1/sessions/{session_id}/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"]
)
async def analyze_session(session_id: str, request: AnalyzeRequest):
    """
    Analyzes student facial video session to return risk indicators (0.0 to 1.0)
    for Depression, Anxiety, and Stress, confidence metrics, top contributing features,
    and human-review flag.
    
    This service is a SCREENING AID, not a clinical diagnostic tool.
    """
    if not request.modalities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one modality must be specified in 'modalities' list."
        )

    results_by_modality: Dict[str, Dict] = {}

    for mod_name in request.modalities:
        if mod_name not in MODALITY_ANALYZERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported modality requested: '{mod_name}'. Supported modalities: {list(MODALITY_ANALYZERS.keys())}"
            )

        analyzer = MODALITY_ANALYZERS[mod_name]

        # Extract modality specific payload from request
        if mod_name == "face":
            if not request.face:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Modality 'face' was requested but 'face' configuration object is missing."
                )
            payload = request.face.model_dump()
        else:
            payload = {}

        try:
            mod_result = await analyzer.analyze(session_id, payload)
            results_by_modality[mod_name] = mod_result
        except FileNotFoundError as e:
            logger.error(f"File not found error during {mod_name} analysis: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error executing {mod_name} analysis for session {session_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error processing {mod_name} modality: {str(e)}"
            )

    # Fuse results from all executed modalities
    response = fusion_engine.combine_results(session_id, results_by_modality)
    return response

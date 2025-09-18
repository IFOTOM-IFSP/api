# app/api/v1/endpoints/analysis.py
from fastapi import APIRouter, HTTPException, Depends
import logging
from app.api.v1.models import (
    AnalysisRequest, AnalysisResponse,
    ReferenceProcessingRequest, ReferenceProcessingResponse
)
from app.core.image_processor import SpectraProcessor
from app.core.dependencies import get_spectra_processor

router = APIRouter()

@router.post("/process-references",
             response_model=ReferenceProcessingResponse,
             summary="Processa dark/white e retorna espectros e métricas de ruído")
async def process_references_endpoint(
    request: ReferenceProcessingRequest,
    processor: SpectraProcessor = Depends(get_spectra_processor)
):
    try:
        return processor.process_references(request)
    except ValueError as ve:
        logging.warning(f"Dados inválidos: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.exception("Erro inesperado")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.post("/analyze",
             response_model=AnalysisResponse,
             summary="Executa análise quantitativa/scan/kinetic")
async def analyze_endpoint(
    request: AnalysisRequest,
    processor: SpectraProcessor = Depends(get_spectra_processor)
):
    try:
        result = processor.run_analysis(request)
        return AnalysisResponse(status="success", results=result)
    except ValueError as ve:
        logging.warning(f"Dados inválidos: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except NotImplementedError as nie:
        raise HTTPException(status_code=501, detail=str(nie))
    except Exception as e:
        logging.exception("Erro inesperado")
        raise HTTPException(status_code=500, detail="Erro interno")

from fastapi import APIRouter, HTTPException, Depends
import logging 
from app.api.v1.models import (
    AnalysisRequest,
    AnalysisResponse,
    ReferenceProcessingRequest,
    ReferenceProcessingResponse
)
from app.core.image_processor import SpectraProcessor 
from app.core.dependencies import get_spectra_processor

router = APIRouter()

@router.post("/process-references", response_model=ReferenceProcessingResponse, summary="Processa e retorna espectros de referência")
async def process_references_endpoint(request: ReferenceProcessingRequest, processor: SpectraProcessor = Depends(get_spectra_processor)):
    try:
        processed_data = processor.process_references(request)
        return ReferenceProcessingResponse(status="success", **processed_data)

    except Exception as e:
        print(f"ERRO NO PROCESSAMENTO DE REFERÊNCIAS: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno do servidor ao processar referências: {e}"
        )

@router.post("/analyze", response_model=AnalysisResponse, summary="Analisa amostras usando referências processadas")
async def analyze_endpoint(request: AnalysisRequest, processor: SpectraProcessor = Depends(get_spectra_processor)):

    try:
        analysis_results = processor.run_analysis(request)

        return AnalysisResponse(status="success", results=analysis_results)

    except ValueError as ve:
        logging.warning(f"Requisição com dados inválidos: {ve}")
        raise HTTPException(
            status_code=400, 
            detail=str(ve)   
        )

    except Exception as e:
        logging.error(f"Erro inesperado no servidor: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Ocorreu um erro interno. A equipe foi notificada."
        )


from fastapi import APIRouter, HTTPException

from app.api.v1.models import (
    AnalysisRequest,
    AnalysisResponse,
    ReferenceProcessingRequest,
    ReferenceProcessingResponse
)
from app.core.image_processor import SpectraProcessor 

router = APIRouter()

@router.post("/process-references", response_model=ReferenceProcessingResponse, summary="Processa e retorna espectros de referência")
async def process_references_endpoint(request: ReferenceProcessingRequest):

    try:
        processor = SpectraProcessor()
        processed_data = processor.process_references(request)
        
        return ReferenceProcessingResponse(status="success", **processed_data)

    except Exception as e:
        # Log do erro detalhado no servidor para depuração.
        print(f"ERRO NO PROCESSAMENTO DE REFERÊNCIAS: {e}")
        # Retorna um erro genérico e seguro para o cliente.
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno do servidor ao processar referências: {e}"
        )

@router.post("/analyze", response_model=AnalysisResponse, summary="Analisa amostras usando referências processadas")
async def analyze_endpoint(request: AnalysisRequest):

    try:
        processor = SpectraProcessor()
        analysis_results = processor.run_analysis(request)

        return AnalysisResponse(status="success", results=analysis_results)

    except Exception as e:
        print(f"ERRO NA ANÁLISE DA AMOSTRA: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno do servidor ao analisar a amostra: {e}"
        )

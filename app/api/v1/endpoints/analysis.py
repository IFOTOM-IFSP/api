
from fastapi import APIRouter, HTTPException
from app.api.v1.models import AnalysisRequest, AnalysisResponse
from app.core.image_processor import SpectraProcessor

router = APIRouter()

@router.post("/process-analysis", response_model=AnalysisResponse)
async def process_analysis_endpoint(request: AnalysisRequest):
    """
    Recebe os dados de imagem da aplicação móvel, invoca o processador
    e retorna os resultados da análise espectral.
    """
    try:
        processor = SpectraProcessor(request)
        results = processor.process()
        
        return AnalysisResponse(status="success", results=results)

    except Exception as e:
        print(f"ERRO NO PROCESSAMENTO: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")

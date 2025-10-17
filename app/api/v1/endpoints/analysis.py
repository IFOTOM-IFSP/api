# app/routes.py
from fastapi import APIRouter
from .schemas import (
    CharacterizeRequest, CharacterizeResponse,
    QuantAnalyzeRequest, QuantAnalyzeResponse
)

router = APIRouter()

# --- Handlers "core" (use seus serviços reais aqui) ---
def process_references_core(req: CharacterizeRequest) -> CharacterizeResponse:
    # TODO: chamar o seu pipeline real
    return CharacterizeResponse(
        status="success",
        dark_reference_spectrum=[[0, 99.0]],
        white_reference_spectrum=[[0, 199.0]],
        dark_current_std_dev=1.2,
        pixel_to_nm=None
    )

def analyze_core(req: QuantAnalyzeRequest) -> QuantAnalyzeResponse:
    # TODO: chamar seu motor real (quant, etc.)
    return QuantAnalyzeResponse(
        status="success",
        results={
            "curve": req.curve.model_dump() if req.curve else None,
            "sample_results": [
                {"sample_absorbance": 0.452, "calculated_concentration": 37.58}
            ],
            "qa": {"notes": "ok"}
        }
    )

# --- Rotas compatíveis com seu app ---
@router.post("/instrument/characterize", response_model=CharacterizeResponse)
def instrument_characterize(req: CharacterizeRequest):
    return process_references_core(req)

@router.post("/quant/analyze", response_model=QuantAnalyzeResponse)
def quant_analyze(req: QuantAnalyzeRequest):
    return analyze_core(req)

# --- Rotas antigas preservadas (retrocompat) ---
@router.post("/api/v1/process-references", response_model=CharacterizeResponse)
def legacy_process_references(req: CharacterizeRequest):
    return process_references_core(req)

@router.post("/api/v1/analyze", response_model=QuantAnalyzeResponse)
def legacy_analyze(req: QuantAnalyzeRequest):
    return analyze_core(req)

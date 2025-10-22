from __future__ import annotations
from fastapi import APIRouter

from .models import (
    CharacterizeRequest, CharacterizeResponse,
    QuantAnalyzeRequest, QuantAnalyzeResponse,
    KineticAnalyzeRequest, KineticAnalyzeResponse,
)
from .dependencies import get_spectra_processor

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
#   Characterize (stub MVP)
# ----------------------------
@router.post("/instrument/characterize", response_model=CharacterizeResponse)
def instrument_characterize(req: CharacterizeRequest):
    # Mantenha stub até integrar seu fluxo real
    return CharacterizeResponse(
        status="success",
        dark_reference_spectrum=[[0, 100.0]],
        white_reference_spectrum=[[0, 2000.0]],
        dark_current_std_dev=1.0,
        pixel_to_nm=None,
    )

# ----------------------------
#   Quant (retrocompat / stub)
# ----------------------------
@router.post("/quant/analyze", response_model=QuantAnalyzeResponse)
def quant_analyze(req: QuantAnalyzeRequest):
    # Mantido como stub para não quebrar seu app atual
    return QuantAnalyzeResponse(status="success", results={"note": "quant stub"})

# ----------------------------
#   Kinetic A(t)
# ----------------------------
@router.post("/kinetic/analyze", response_model=KineticAnalyzeResponse)
def kinetic_analyze(req: KineticAnalyzeRequest):
    sp = get_spectra_processor()
    return sp.run_kinetic(req)

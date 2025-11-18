from __future__ import annotations
from fastapi import APIRouter

from .models import (
    CharacterizeRequest, CharacterizeResponse,
    QuantAnalyzeRequest, QuantAnalyzeResponse,
)
from .dependencies import get_spectra_processor

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/instrument/characterize", response_model=CharacterizeResponse)
def instrument_characterize(req: CharacterizeRequest):
    """
    Calibração do instrumento com:
      - dark_burst
      - white_burst (opcional)
      - 2 lasers (verde e vermelho)

    Retorna:
      - dark_reference_spectrum (médio)
      - white_reference_spectrum (médio, se fornecido)
      - dark_current_std_dev
      - pixel_to_nm (a0, a1, a2)
    """
    sp = get_spectra_processor()
    return sp.run_characterize(req)

@router.post("/analysis/quant", response_model=QuantAnalyzeResponse)
@router.post("/quant/analyze", response_model=QuantAnalyzeResponse) 
def quant_analyze(req: QuantAnalyzeRequest):
    sp = get_spectra_processor()
    return sp.run_quantitative(req)

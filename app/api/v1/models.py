

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple

# --- 1. Modelos de Resposta  ---


class CalibrationCurve(BaseModel):
    r_squared: float
    equation: str

class AnalysisResult(BaseModel):
    calibration_curve: Optional[CalibrationCurve] = None
    calculated_concentration: Optional[float] = None
    sample_absorbance: Optional[float] = None
    spectrum_data: Optional[List[Tuple[float, float]]] = Field(None, example=[(400.5, 0.1), (401.0, 0.12)])

class AnalysisResponse(BaseModel):
    status: Literal['success', 'error']
    results: Optional[AnalysisResult] = None
    error: Optional[str] = None

# --- 2. Modelos de Pedido ---


class SampleFrame(BaseModel):
    type: Literal['standard', 'unknown']
    concentration: Optional[float] = None
    frames_base64: List[str]
    target_wavelength_range: Optional[Tuple[float, float]] = None

class AnalysisRequest(BaseModel):
    analysisType: Literal['quantitative', 'scan', 'kinetic']
    dark_frames_base64: List[str]
    white_frames_base64: List[str]
    samples: List[SampleFrame]
    calibration_coefficients: Optional[List[float]] = None
    known_wavelengths_for_calibration: Optional[List[float]] = Field(None, example=[465, 545])

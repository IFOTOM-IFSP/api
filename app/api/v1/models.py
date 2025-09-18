from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int

class SpectralBurst(BaseModel):
    """De preferência, receba vetores 1D por frame (já somados por coluna na ROI no dispositivo)."""
    vectors: List[List[float]] = Field(..., description="[frames][numPixels]")
    timestamps_ms: Optional[List[int]] = None

class ImageBurst(BaseModel):
    """Fallback se ainda precisar mandar imagens base64."""
    frames_base64: List[str]
    timestamps_ms: Optional[List[int]] = None

class CalibrationCurveInput(BaseModel):
    slope: float
    intercept: float
    r_squared: Optional[float] = None
    s_m: Optional[float] = None
    s_b: Optional[float] = None
    lod: Optional[float] = None
    loq: Optional[float] = None

class PixelToWavelength(BaseModel):
    coeffs: List[float] = Field(..., description="a0,a1[,a2] para λ = a0 + a1*px + a2*px²")
    rmse_nm: Optional[float] = None
    dispersion_nm_per_px: Optional[float] = None

class ReferenceProcessingRequest(BaseModel):
    dark: SpectralBurst | ImageBurst
    white: SpectralBurst | ImageBurst
    roi: Optional[ROI] = None

class ReferenceProcessingResponse(BaseModel):
    status: Literal['success', 'error']
    dark_reference_spectrum: Optional[List[Tuple[int, float]]] = None   # [(px, I_dark)]
    white_reference_spectrum: Optional[List[Tuple[int, float]]] = None  # [(px, I_white)]
    pixel_to_wavelength: Optional[PixelToWavelength] = None
    dark_current_std_dev: Optional[float] = None
    error: Optional[str] = None

class SampleInput(BaseModel):
    kind: Literal['standard','unknown']
    burst: SpectralBurst | ImageBurst
    concentration: Optional[float] = None
    dilution_factor: float = 1.0

class AnalysisRequest(BaseModel):
    analysisType: Literal['quantitative','simple_read','scan','kinetic']
    pixel_to_wavelength: Optional[PixelToWavelength] = None
    calibration_curve: Optional[CalibrationCurveInput] = None
    target_wavelength: Optional[float] = Field(None, description="λ de leitura em nm")
    window_nm: float = Field(4.0, description="meia-largura da janela espectral ±nm")
    roi: Optional[ROI] = None
    dark_reference_spectrum: List[Tuple[int,float]]
    white_reference_spectrum: List[Tuple[int,float]]
    samples: List[SampleInput]

class SampleResult(BaseModel):
    sample_absorbance: float
    calculated_concentration: Optional[float] = None
    spectrum_data: List[Tuple[float, float]]  # [(λ, A(λ))]

class CalibrationCurve(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    equation: str
    see: Optional[float] = None

class AnalysisResult(BaseModel):
    calibration_curve: Optional[CalibrationCurve] = None
    sample_results: List[SampleResult]
    qa: Optional[dict] = None

class AnalysisResponse(BaseModel):
    status: Literal['success','error']
    results: Optional[AnalysisResult] = None
    error: Optional[str] = None

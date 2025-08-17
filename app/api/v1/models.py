from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator

class SampleFrame(BaseModel):
    type: Literal['standard', 'unknown']
    frames_base64: List[str]
    
    concentration: Optional[float] = Field(None, description="Concentração do padrão de calibração.")
    
    dilution_factor: float = Field(1.0, description="Fator de diluição aplicado à amostra.")

    timestamps_sec: Optional[List[float]] = Field(None, description="Lista de timestamps em segundos para cada frame, para análise cinética.")

class CalibrationCurveInput(BaseModel):
    slope: float
    intercept: float

class AnalysisRequest(BaseModel):
    analysisType: Literal['quantitative', 'scan', 'simple_read', 'kinetic']
    calibration_curve: Optional[CalibrationCurveInput] = Field(None, description="Parâmetros (slope, intercept) de uma curva de calibração existente.")
    dark_reference_spectrum: List[Tuple[int, float]]
    white_reference_spectrum: List[Tuple[int, float]]
    pixel_to_wavelength_coeffs: List[float]
    
    samples: List[SampleFrame]
    
    target_wavelength: Optional[float] = Field(None, description="Comprimento de onda alvo para 'quantitative', 'simple_read' e 'kinetic'")
    scan_range: Optional[Tuple[float, float]] = Field(None, description="Intervalo (início, fim) em nm para 'scan'")
    optical_path_cm: float = Field(1.0, description="Caminho óptico da cubeta em cm")
    @model_validator(mode='after')
    def check_calibration_method(self) -> 'AnalysisRequest':
        has_existing_curve = self.calibration_curve is not None
        has_standard_samples = any(s.type == 'standard' for s in self.samples)

        if self.analysisType == 'quantitative':
            if has_existing_curve and has_standard_samples:
                raise ValueError("Forneça os parâmetros de 'calibration_curve' OU amostras 'standard', mas não ambos.")
            
            if not has_existing_curve and not has_standard_samples:
                raise ValueError("Análise quantitativa requer os parâmetros de 'calibration_curve' ou amostras 'standard'.")
                
        return self

class ReferenceProcessingRequest(BaseModel):
    dark_frames_base64: List[str]
    white_frames_base64: List[str]
    known_wavelengths_for_calibration: Optional[List[float]] = Field(None, example=[465, 545])
    peak_detection_height_factor: Optional[float] = Field(1.1, description="Multiplicador da média para altura mínima do pico.")
    peak_detection_distance: Optional[int] = Field(50, description="Distância mínima em pixels entre os picos.")

class CalibrationCurve(BaseModel):
    r_squared: float
    equation: str
    slope: float
    intercept: float

class SampleResult(BaseModel):
    sample_absorbance: Optional[float] = None
    calculated_concentration: Optional[float] = None
    spectrum_data: Optional[List[Tuple[float, float]]] = Field(None, description="Espectro (comprimento de onda, absorbância)")

    kinetic_data: Optional[List[Tuple[float, float]]] = Field(None, description="Dados cinéticos (tempo em seg, absorbância)")


class AnalysisResult(BaseModel):
    calibration_curve: Optional[CalibrationCurve] = None
    sample_results: List[SampleResult]


class AnalysisResponse(BaseModel):
    status: Literal['success', 'error']
    results: Optional[AnalysisResult] = None
    error: Optional[str] = None


class ReferenceProcessingResponse(BaseModel):
    status: Literal['success', 'error']
    dark_reference_spectrum: Optional[List[Tuple[int, float]]] = None
    white_reference_spectrum: Optional[List[Tuple[int, float]]] = None
    pixel_to_wavelength_coeffs: Optional[List[float]] = None
    dark_current_std_dev: Optional[float] = None
    error: Optional[str] = None


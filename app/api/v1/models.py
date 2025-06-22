from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple


class SampleFrame(BaseModel):
    """
    Representa uma única amostra (padrão ou desconhecida) a ser analisada.
    """
    type: Literal['standard', 'unknown']
    frames_base64: List[str]
    
    concentration: Optional[float] = Field(None, description="Concentração do padrão de calibração (obrigatório se type='standard')")
    
    dilution_factor: float = Field(1.0, description="Fator de diluição aplicado à amostra (padrão é 1, sem diluição)")


class AnalysisRequest(BaseModel):
    """
    Modelo para o endpoint /analyze. Recebe as referências já processadas
    e os dados das amostras para calcular o resultado final.
    """
    analysisType: Literal['quantitative', 'scan', 'simple_read']
    
    dark_reference_spectrum: List[Tuple[int, float]] = Field(..., description="Espectro de referência do escuro (pixel, intensidade)")
    white_reference_spectrum: List[Tuple[int, float]] = Field(..., description="Espectro de referência do branco (pixel, intensidade)")
    
    pixel_to_wavelength_coeffs: List[float] = Field(..., description="Coeficientes do polinômio de calibração de comprimento de onda")
    

    samples: List[SampleFrame]
    

    target_wavelength: Optional[float] = Field(None, description="Comprimento de onda alvo para 'quantitative' e 'simple_read'")
    scan_range: Optional[Tuple[float, float]] = Field(None, description="Intervalo (início, fim) para 'scan'")
    optical_path_cm: float = Field(1.0, description="Caminho óptico da cubeta em cm")



class ReferenceProcessingRequest(BaseModel):
    dark_frames_base64: List[str]
    white_frames_base64: List[str]
    
   
    known_wavelengths_for_calibration: Optional[List[float]] = Field(None, example=[465, 545])




class CalibrationCurve(BaseModel):
    r_squared: float
    equation: str 
    slope: float
    intercept: float


class SampleResult(BaseModel):
    """Contém o resultado da análise para uma única amostra desconhecida."""
    sample_absorbance: Optional[float] = None
    calculated_concentration: Optional[float] = None
    spectrum_data: Optional[List[Tuple[float, float]]] = Field(None, description="Espectro completo (comprimento de onda, absorbância)")


class AnalysisResult(BaseModel):
    """Agrega todos os resultados de uma chamada de análise."""
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

# app/api/v1/models.py

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple

# =============================================================================
# 1. MODELOS DE PEDIDO (REQUEST MODELS)
# =============================================================================

class SampleFrame(BaseModel):
    """
    Representa uma única amostra a ser analisada. Adaptado para todos os modos.
    """
    type: Literal['standard', 'unknown']
    frames_base64: List[str]
    
    # Campo para padrões de calibração
    concentration: Optional[float] = Field(None, description="Concentração do padrão de calibração.")
    
    # Campo para amostras desconhecidas (quantitativa)
    dilution_factor: float = Field(1.0, description="Fator de diluição aplicado à amostra.")

    # --> NOVO: Campo para análise cinética, com os tempos de cada frame
    timestamps_sec: Optional[List[float]] = Field(None, description="Lista de timestamps em segundos para cada frame, para análise cinética.")


class AnalysisRequest(BaseModel):
    """
    Modelo para o endpoint /analyze.
    """
    # --> MUDANÇA: 'kinetic' foi adicionado novamente aos tipos de análise suportados.
    analysisType: Literal['quantitative', 'scan', 'simple_read', 'kinetic']
    
    dark_reference_spectrum: List[Tuple[int, float]]
    white_reference_spectrum: List[Tuple[int, float]]
    pixel_to_wavelength_coeffs: List[float]
    
    samples: List[SampleFrame]
    
    # Parâmetros específicos por tipo de análise
    target_wavelength: Optional[float] = Field(None, description="Comprimento de onda alvo para 'quantitative', 'simple_read' e 'kinetic'")
    scan_range: Optional[Tuple[float, float]] = Field(None, description="Intervalo (início, fim) em nm para 'scan'")
    optical_path_cm: float = Field(1.0, description="Caminho óptico da cubeta em cm")


class ReferenceProcessingRequest(BaseModel):
    """Modelo para o endpoint /process-references."""
    dark_frames_base64: List[str]
    white_frames_base64: List[str]
    known_wavelengths_for_calibration: Optional[List[float]] = Field(None, example=[465, 545])


# =============================================================================
# 2. MODELOS DE RESPOSTA (RESPONSE MODELS)
# =============================================================================

class CalibrationCurve(BaseModel):
    r_squared: float
    equation: str
    slope: float
    intercept: float

class SampleResult(BaseModel):
    """Contém o resultado da análise para uma única amostra."""
    # Para modos quantitativo e scan
    sample_absorbance: Optional[float] = None
    calculated_concentration: Optional[float] = None
    spectrum_data: Optional[List[Tuple[float, float]]] = Field(None, description="Espectro (comprimento de onda, absorbância)")

    # --> NOVO: Campo para armazenar o resultado de uma análise cinética
    kinetic_data: Optional[List[Tuple[float, float]]] = Field(None, description="Dados cinéticos (tempo em seg, absorbância)")


class AnalysisResult(BaseModel):
    """Agrega todos os resultados de uma chamada de análise."""
    calibration_curve: Optional[CalibrationCurve] = None
    sample_results: List[SampleResult]


class AnalysisResponse(BaseModel):
    status: Literal['success', 'error']
    results: Optional[AnalysisResult] = None
    error: Optional[str] = None


class ReferenceProcessingResponse(BaseModel):
    """Modelo de resposta para /process-references."""
    status: Literal['success', 'error']
    dark_reference_spectrum: Optional[List[Tuple[int, float]]] = None
    white_reference_spectrum: Optional[List[Tuple[int, float]]] = None
    pixel_to_wavelength_coeffs: Optional[List[float]] = None
    dark_current_std_dev: Optional[float] = None
    error: Optional[str] = None

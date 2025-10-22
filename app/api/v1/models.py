from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# =====================
#   Tipos básicos
# =====================

class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int

class PixelToNm(BaseModel):
    """Coeficientes do polinômio pixel→nm: λ = a0 + a1·p + a2·p²
    a2 pode ser 0.0 para ajuste linear.
    """
    a0: float
    a1: float
    a2: float = 0.0

class Curve(BaseModel):
    """Curva de calibração A = m·C + b"""
    m: float
    b: float

class ReferenceBurst(BaseModel):
    """Conjunto de espectros 1D (pós-ROI, já colapsados por coluna).
    Cada item em `vectors` é um espectro de uma aquisição.
    """
    vectors: List[List[float]]

# =====================
#   Cinética A(t)
# =====================

class TimePoint(BaseModel):
    t_sec: float
    burst: ReferenceBurst

class KineticFitSpec(BaseModel):
    model: Literal["first_order", "second_order", "none"] = "first_order"
    baseline_strategy: Literal["tail_median", "fit_param"] = "tail_median"
    tail_fraction: float = 0.2

class KineticAnalyzeRequest(BaseModel):
    """Payload para análise cinética A(t) em λ fixo.
    - dark/white: espectros de referência (mesmo comprimento dos vetores das séries)
    - series: lista de pontos no tempo com bursts.
    - pixel_to_nm: opcional (alias `pixel_to_wavelength` aceito para retrocompat).
    """
    analysisType: Literal["kinetic"] = "kinetic"

    pixel_to_nm: Optional[PixelToNm] = Field(None, alias="pixel_to_wavelength")
    target_wavelength: float
    window_nm: float = 4.0
    roi: Optional[ROI] = None

    dark_reference_spectrum: List[List[float]]
    white_reference_spectrum: List[List[float]]

    series: List[TimePoint]
    fit: Optional[KineticFitSpec] = None

    model_config = ConfigDict(populate_by_name=True)

class KineticAnalyzeResponse(BaseModel):
    status: Literal["success", "error"]
    results: Optional[dict] = None
    error: Optional[str] = None

# =====================
#   Outros (existentes / retrocompat)
# =====================

class CharacterizeRequest(BaseModel):
    # MVP simples (mantenha/expanda conforme o seu fluxo de caracterização)
    pass

class CharacterizeResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    dark_reference_spectrum: Optional[List[List[float]]] = None
    white_reference_spectrum: Optional[List[List[float]]] = None
    dark_current_std_dev: Optional[float] = None
    pixel_to_nm: Optional[PixelToNm] = None

class QuantAnalyzeRequest(BaseModel):
    """Mantido aqui para compatibilidade com sua rota /quant/analyze.
    Caso você já tenha outro shape no seu projeto, mantenha o seu e ignore este.
    """
    analysisType: Literal["quantitative", "simple_read", "scan", "kinetic"]
    pixel_to_nm: Optional[PixelToNm] = Field(None, alias="pixel_to_wavelength")
    roi: Optional[ROI] = None

    # Referências
    dark_reference_spectrum: Optional[List[List[float]]] = None
    white_reference_spectrum: Optional[List[List[float]]] = None

    # Dados para leitura simples/quant (ex.: um burst só)
    burst: Optional[ReferenceBurst] = None

    # Curva (para quantitative)
    curve: Optional[Curve] = None

    target_wavelength: Optional[float] = None
    window_nm: Optional[float] = 4.0

    model_config = ConfigDict(populate_by_name=True)

class QuantAnalyzeResponse(BaseModel):
    status: Literal["success", "error"]
    results: Optional[dict] = None
    error: Optional[str] = None

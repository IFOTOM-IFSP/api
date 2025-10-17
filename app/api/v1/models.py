# app/schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

class ROI(BaseModel):
    x: int
    y: int
    w: int
    h: int

class PixelToNm(BaseModel):
    a0: float
    a1: float
    a2: Optional[float] = None
    rmse_nm: Optional[float] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    def accept_legacy_coeffs(cls, v):
        # aceita {"coeffs":[a0,a1,a2?], "rmse_nm":...}
        if isinstance(v, dict) and "coeffs" in v and ("a0" not in v and "a1" not in v):
            coeffs = v.get("coeffs") or []
            return {
                "a0": coeffs[0] if len(coeffs) > 0 else None,
                "a1": coeffs[1] if len(coeffs) > 1 else None,
                "a2": coeffs[2] if len(coeffs) > 2 else None,
                "rmse_nm": v.get("rmse_nm"),
            }
        return v

class Curve(BaseModel):
    m: float
    b: float
    R2: Optional[float] = Field(None, description="coeficiente de determinação")

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    def accept_legacy_curve(cls, v):
        # aceita {slope, intercept, r_squared}
        if isinstance(v, dict) and "m" not in v and "b" not in v:
            return {
                "m": v.get("slope"),
                "b": v.get("intercept"),
                "R2": v.get("r_squared") or v.get("R2"),
            }
        return v

class ReferenceBurst(BaseModel):
    vectors: List[List[float]]

class SampleBurst(BaseModel):
    kind: Literal["unknown","standard","blank"] = "unknown"
    burst: ReferenceBurst

class CharacterizeRequest(BaseModel):
    dark: ReferenceBurst
    white: ReferenceBurst
    roi: Optional[ROI] = None

class CharacterizeResponse(BaseModel):
    status: Literal["success"]
    dark_reference_spectrum: List[List[float]]
    white_reference_spectrum: List[List[float]]
    dark_current_std_dev: float
    pixel_to_nm: Optional[PixelToNm] = None  # responder já no novo nome

class QuantAnalyzeRequest(BaseModel):
    analysisType: Literal["quantitative","simple_read","scan","kinetic"] = "quantitative"
    pixel_to_nm: Optional[PixelToNm] = Field(None, alias="pixel_to_wavelength")
    curve: Optional[Curve] = Field(None, alias="calibration_curve")
    target_wavelength: float
    window_nm: float = 4.0
    roi: Optional[ROI] = None
    dark_reference_spectrum: List[List[float]]
    white_reference_spectrum: List[List[float]]
    samples: List[SampleBurst]

    model_config = ConfigDict(populate_by_name=True)

class SampleResult(BaseModel):
    sample_absorbance: float
    calculated_concentration: Optional[float] = None
    spectrum_data: Optional[List[List[float]]] = None

class QuantAnalyzeResponse(BaseModel):
    status: Literal["success"]
    results: dict

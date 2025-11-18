from pydantic import BaseModel
from typing import List, Optional


class ROI(BaseModel):
    x0: int
    x1: int
    y0: int
    y1: int


class PixelToNm(BaseModel):
    a0: float 
    a1: float 
    a2: float = 0.0 


class Curve(BaseModel):
    m: float
    b: float
    r2: Optional[float] = None


class ReferenceBurst(BaseModel):
    vectors: List[List[float]]

class LaserCalibBurst(BaseModel):
    tag: str              
    lambda_nm: float      
    burst: ReferenceBurst

class CharacterizeRequest(BaseModel):
    device_id: str
    roi: Optional[ROI] = None
    dark_burst: ReferenceBurst
    white_burst: Optional[ReferenceBurst] = None
    lasers: List[LaserCalibBurst]   
    metadata: dict = {}


class CharacterizeResponse(BaseModel):
    status: str
    dark_reference_spectrum: List[List[float]]      
    white_reference_spectrum: Optional[List[List[float]]] = None
    dark_current_std_dev: float
    pixel_to_nm: Optional[PixelToNm] = None
    error: Optional[str] = None

class QuantAnalyzeRequest(BaseModel):
    dark_reference_spectrum: List[float] | List[List[float]]
    white_reference_spectrum: List[float] | List[List[float]]
    sample_bursts: List[ReferenceBurst]
    roi: Optional[ROI] = None
    pixel_to_nm: Optional[PixelToNm] = None
    target_wavelength: float
    window_nm: float = 4.0
    curve: Curve


class QuantAnalyzeResponse(BaseModel):
    status: str
    results: Optional[dict] = None
    error: Optional[str] = None

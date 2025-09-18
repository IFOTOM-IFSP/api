from __future__ import annotations
import base64
import numpy as np
import cv2
from typing import List, Tuple, Optional

from app.api.v1.models import (
    ReferenceProcessingRequest, ReferenceProcessingResponse,
    AnalysisRequest, AnalysisResult, SampleResult, ROI
)


class SpectraProcessor:
    """
    Núcleo da API. Mantém tudo stateless: cada chamada traz dados brutos e devolve resultados.
    """

    def _base64_to_image(self, base64_string: str) -> np.ndarray:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # já em Y (luma)
        if image is None:
            raise ValueError("Frame base64 inválido")
        return image

    def _profile_from_image(self, image: np.ndarray, roi: Optional[ROI]) -> np.ndarray:
        if roi:
            x, y, w, h = roi.x, roi.y, roi.w, roi.h
            image = image[y:y+h, x:x+w]
        prof = image.sum(axis=0).astype(np.float64)
        return prof

    def _robust_mean_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float64)  # [frames, numPx]
        if arr.ndim != 2:
            raise ValueError("vectors deve ser 2D [frames, numPixels]")
        return np.median(arr, axis=0)

    def _avg_profile_from_base64_burst(self, frames_b64: List[str], roi: Optional[ROI]) -> np.ndarray:
        profiles = [self._profile_from_image(self._base64_to_image(b64), roi) for b64 in frames_b64]
        return np.median(np.stack(profiles, axis=0), axis=0)

    def _compensate_spectrum(self, sample: np.ndarray, dark: np.ndarray, white: np.ndarray) -> np.ndarray:
        L = int(min(len(sample), len(dark), len(white)))
        s = np.clip(sample[:L] - dark[:L], 0, None)
        r = np.clip(white[:L] - dark[:L], 1e-9, None)
        T = s / r
        T = np.clip(T, 1e-6, 1.0)
        A = -np.log10(T)
        return A

    def _px_to_nm(self, length: int, coeffs: List[float]) -> np.ndarray:
        px = np.arange(length, dtype=np.float64)
        coeffs_polyval = list(reversed(coeffs))
        return np.polyval(coeffs_polyval, px)

    def _abs_at_lambda(self, wavelengths: np.ndarray, A: np.ndarray, lambda0: float, window_nm: float) -> float:
        mask = (wavelengths >= lambda0 - window_nm) & (wavelengths <= lambda0 + window_nm)
        if not np.any(mask):
            raise ValueError("Janela espectral vazia para o λ alvo")
        return float(np.mean(A[mask]))

    def process_references(self, request: ReferenceProcessingRequest) -> ReferenceProcessingResponse:
        if hasattr(request.dark, "vectors"):
            dark_vec = self._robust_mean_vectors(request.dark.vectors)
            dark_std = float(np.std(np.asarray(request.dark.vectors), axis=0).mean())
        else:
            dark_vec = self._avg_profile_from_base64_burst(request.dark.frames_base64, request.roi)
            dark_std = float(np.std(dark_vec))

        if hasattr(request.white, "vectors"):
            white_vec = self._robust_mean_vectors(request.white.vectors)
        else:
            white_vec = self._avg_profile_from_base64_burst(request.white.frames_base64, request.roi)

        L = int(min(len(dark_vec), len(white_vec)))
        dark_vec = dark_vec[:L]
        white_vec = white_vec[:L]

        dark_pairs = list(zip(range(L), dark_vec.tolist()))
        white_pairs = list(zip(range(L), white_vec.tolist()))

        return ReferenceProcessingResponse(
            status="success",
            dark_reference_spectrum=dark_pairs,
            white_reference_spectrum=white_pairs,
            dark_current_std_dev=dark_std,
            pixel_to_wavelength=None,
        )

    def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        atype = request.analysisType

        dark_px, dark_I = zip(*request.dark_reference_spectrum)
        white_px, white_I = zip(*request.white_reference_spectrum)
        dark = np.asarray(dark_I, dtype=np.float64)
        white = np.asarray(white_I, dtype=np.float64)
        L = int(min(len(dark), len(white)))
        dark, white = dark[:L], white[:L]

        wavelengths = np.arange(L, dtype=np.float64)
        if request.pixel_to_wavelength:
            wavelengths = self._px_to_nm(L, request.pixel_to_wavelength.coeffs)

        if atype == "quantitative" or atype == "simple_read":
            return self._analyze_quantitative_like(request, dark, white, wavelengths)
        elif atype == "scan":
            return self._analyze_scan(request, dark, white, wavelengths)
        elif atype == "kinetic":
            # implementar no futuro (variação temporal)
            raise NotImplementedError("Kinetic ainda não implementado.")
        else:
            raise ValueError(f"analysisType inválido: {atype}")

    def _get_burst_mean_vector(self, burst, roi: Optional[ROI]) -> np.ndarray:
        if hasattr(burst, "vectors"):
            return self._robust_mean_vectors(burst.vectors)
        else:
            return self._avg_profile_from_base64_burst(burst.frames_base64, roi)

    def _analyze_quantitative_like(
        self,
        request: AnalysisRequest,
        dark: np.ndarray,
        white: np.ndarray,
        wavelengths: np.ndarray
    ) -> AnalysisResult:
        if request.target_wavelength is None:
            raise ValueError("Para quantitative/simple_read informe target_wavelength (nm).")

        sample_results: List[SampleResult] = []
        for s in request.samples:
            vec = self._get_burst_mean_vector(s.burst, request.roi)[:len(dark)]
            A = self._compensate_spectrum(vec, dark, white)

            A0 = self._abs_at_lambda(wavelengths, A, request.target_wavelength, request.window_nm)

            C = None
            if request.calibration_curve and request.calibration_curve.slope != 0:
                C = (A0 - request.calibration_curve.intercept) / request.calibration_curve.slope
                C *= s.dilution_factor if s.dilution_factor else 1.0

            sample_results.append(
                SampleResult(
                    sample_absorbance=A0,
                    calculated_concentration=C,
                    spectrum_data=list(zip(wavelengths.tolist(), A.tolist()))
                )
            )

        return AnalysisResult(
            calibration_curve=None,
            sample_results=sample_results,
            qa={"notes": "ok"}
        )

    def _analyze_scan(
        self,
        request: AnalysisRequest,
        dark: np.ndarray,
        white: np.ndarray,
        wavelengths: np.ndarray
    ) -> AnalysisResult:
        sample_results: List[SampleResult] = []
        for s in request.samples:
            vec = self._get_burst_mean_vector(s.burst, request.roi)[:len(dark)]
            A = self._compensate_spectrum(vec, dark, white)
            sample_results.append(
                SampleResult(
                    sample_absorbance=float(np.max(A)),
                    calculated_concentration=None,
                    spectrum_data=list(zip(wavelengths.tolist(), A.tolist()))
                )
            )
        return AnalysisResult(
            calibration_curve=None,
            sample_results=sample_results,
            qa={"notes": "scan"}
        )

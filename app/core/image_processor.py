from __future__ import annotations
from typing import List, Optional

import numpy as np

from .models import (
    ROI,
    PixelToNm,
    Curve,
    ReferenceBurst,
    LaserCalibBurst,
    CharacterizeRequest,
    CharacterizeResponse,
    QuantAnalyzeRequest,
    QuantAnalyzeResponse,
)

def _as_1d(y):
    """Aceita [[i, val], ...] ou [val, val, ...] e retorna np.ndarray 1D de valores."""
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1].astype(float)
    return arr.astype(float)


class SpectraProcessor:
    """
    Core espectral simplificado para IFOTOM.

    Objetivo: pegar dark/white + bursts de amostra e devolver:
      - absorbância média em λ alvo
      - desvio, CV
      - concentração via curva A = m*C + b
    """

    def _get_burst_mean_vector(
        self,
        burst: ReferenceBurst,
        roi: Optional[ROI],
    ) -> np.ndarray:
        """
        Retorna a média dos vetores do burst.

        Aqui assumimos que os vetores já são espectros 1D pós-ROI.
        Se quiser aplicar ROI em imagem, isso deve acontecer antes.
        """
        vectors = [np.asarray(v, dtype=float) for v in burst.vectors]
        if not vectors:
            raise ValueError("Burst sem vetores")
        L = min(v.shape[0] for v in vectors)
        stack = np.vstack([v[:L] for v in vectors])
        return stack.mean(axis=0)

    def _compensate_spectrum(
        self,
        raw: np.ndarray,
        dark: np.ndarray,
        white: np.ndarray,
    ) -> np.ndarray:
        """
        Compensa dark/white e devolve absorbância A = -log10(T).
        """
        eps = 1e-6
        denom = np.clip(white - dark, eps, None)
        T = np.clip((raw - dark) / denom, eps, 1.0)
        A = -np.log10(T)
        return A

    def _px_to_nm(self, length: int, coeffs: List[float]) -> np.ndarray:
        p = np.arange(length, dtype=float)
        a0 = coeffs[0] if len(coeffs) > 0 else 0.0
        a1 = coeffs[1] if len(coeffs) > 1 else 1.0
        a2 = coeffs[2] if len(coeffs) > 2 else 0.0
        return a0 + a1 * p + a2 * (p ** 2)

    def _abs_at_lambda(
        self,
        wavelengths: Optional[np.ndarray],
        A: np.ndarray,
        lambda0: float,
        window_nm: float,
    ) -> float:
        """
        Retorna a absorbância média em torno de lambda0 (janela window_nm).
        Se não houver calibração pixel->nm, assume lambda0 como índice de pixel.
        """
        if wavelengths is None:
            idx = int(round(lambda0))
            idx = max(0, min(idx, len(A) - 1))
            return float(A[idx])

        mask = np.abs(wavelengths - lambda0) <= (window_nm / 2.0)
        if not np.any(mask):
            # fallback: ponto mais próximo
            idx = int(np.argmin(np.abs(wavelengths - lambda0)))
            return float(A[idx])
        return float(A[mask].mean())
        
    def _find_peak_pixel(
        self,
        intensities: np.ndarray,
        dark: np.ndarray,
        window: int = 5,
    ) -> float:
        """
        Subtrai dark, encontra o pixel de maior intensidade e calcula
        o centroide numa janela em torno do pico.
        """
        corrected = intensities - dark
        corrected[corrected < 0] = 0

        p_max = int(np.argmax(corrected))
        start = max(p_max - window, 0)
        end = min(p_max + window + 1, len(corrected))

        x = np.arange(start, end)
        y = corrected[start:end]

        if y.sum() <= 0:
            return float(p_max)

        p_centroid = float((x * y).sum() / y.sum())
        return p_centroid

    def _characterize_instrument(
        self,
        request: CharacterizeRequest,
    ) -> dict:
        dark_vecs = [np.asarray(v, dtype=float) for v in request.dark_burst.vectors]
        if not dark_vecs:
            raise ValueError("dark_burst sem vetores")

        L = min(v.shape[0] for v in dark_vecs)
        dark_stack = np.vstack([v[:L] for v in dark_vecs])
        dark_mean = dark_stack.mean(axis=0)
        dark_std = dark_stack.std(axis=0, ddof=1) if dark_stack.shape[0] > 1 else np.zeros_like(dark_mean)
        dark_std_global = float(dark_std.mean())

        dark_ref = [[int(i), float(val)] for i, val in enumerate(dark_mean.tolist())]

        white_ref = None
        if request.white_burst is not None:
            white_vecs = [np.asarray(v, dtype=float) for v in request.white_burst.vectors]
            if white_vecs:
                Lw = min(v.shape[0] for v in white_vecs)
                white_stack = np.vstack([v[:Lw] for v in white_vecs])
                white_mean = white_stack.mean(axis=0)
                white_ref = [[int(i), float(val)] for i, val in enumerate(white_mean.tolist())]
            else:
                white_mean = None
        else:
            white_mean = None

        if not request.lasers or len(request.lasers) < 2:
            raise ValueError("É necessário fornecer pelo menos 2 lasers para calibração.")

        lasers_sorted: List[LaserCalibBurst] = sorted(request.lasers, key=lambda L: L.lambda_nm)

        laser_low = lasers_sorted[0]
        laser_high = lasers_sorted[-1]

        dark_for_laser = dark_mean

        def mean_vec_from_burst(burst: ReferenceBurst) -> np.ndarray:
            vecs = [np.asarray(v, dtype=float) for v in burst.vectors]
            if not vecs:
                raise ValueError("Burst de laser sem vetores")
            Lb = min(v.shape[0] for v in vecs)
            stack = np.vstack([v[:Lb] for v in vecs])
            return stack.mean(axis=0)

        spec_low = mean_vec_from_burst(laser_low.burst)
        spec_high = mean_vec_from_burst(laser_high.burst)

        pg = self._find_peak_pixel(spec_low, dark_for_laser)
        ph = self._find_peak_pixel(spec_high, dark_for_laser)

        lambda_low = float(laser_low.lambda_nm)
        lambda_high = float(laser_high.lambda_nm)

        if ph == pg:
            raise ValueError("Picos de laser caíram no mesmo pixel; verifique o arranjo óptico.")

        a1 = (lambda_high - lambda_low) / (ph - pg)
        a0 = lambda_low - a1 * pg
        a2 = 0.0

        pixel_to_nm = PixelToNm(a0=a0, a1=a1, a2=a2)

        results = dict(
            dark_reference_spectrum=dark_ref,
            white_reference_spectrum=white_ref,
            dark_current_std_dev=dark_std_global,
            pixel_to_nm=pixel_to_nm,
            meta=dict(
                device_id=request.device_id,
                lasers_used=[
                    dict(tag=laser_low.tag, lambda_nm=lambda_low, pixel_peak=pg),
                    dict(tag=laser_high.tag, lambda_nm=lambda_high, pixel_peak=ph),
                ],
            ),
        )
        return results

    def run_characterize(self, request: CharacterizeRequest) -> CharacterizeResponse:
        try:
            res = self._characterize_instrument(request)
            return CharacterizeResponse(
                status="success",
                dark_reference_spectrum=res["dark_reference_spectrum"],
                white_reference_spectrum=res["white_reference_spectrum"],
                dark_current_std_dev=res["dark_current_std_dev"],
                pixel_to_nm=res["pixel_to_nm"],
            )
        except Exception as e:
            return CharacterizeResponse(
                status="error",
                dark_reference_spectrum=[],
                white_reference_spectrum=None,
                dark_current_std_dev=0.0,
                pixel_to_nm=None,
                error=str(e),
            )

    def _analyze_quantitative(
        self,
        request: QuantAnalyzeRequest,
    ) -> dict:
        """
        Fluxo:
          - dark/white -> espectro A(λ) para cada burst de amostra
          - extrai A em λ alvo
          - faz média, desvio, CV
          - converte para concentração usando curva A = m*C + b
        """
        dark = _as_1d(request.dark_reference_spectrum)
        white = _as_1d(request.white_reference_spectrum)

        L = len(dark)

        wavelengths = None
        if request.pixel_to_nm is not None:
            coeffs = [request.pixel_to_nm.a0, request.pixel_to_nm.a1, request.pixel_to_nm.a2]
            wavelengths = self._px_to_nm(L, coeffs)

        A_vals: List[float] = []
        for burst in request.sample_bursts:
            vec = self._get_burst_mean_vector(burst, request.roi)[:L]
            A_spec = self._compensate_spectrum(vec, dark, white)
            A0 = self._abs_at_lambda(
                wavelengths=wavelengths,
                A=A_spec,
                lambda0=request.target_wavelength,
                window_nm=request.window_nm,
            )
            A_vals.append(float(A0))

        if not A_vals:
            raise ValueError("Nenhum burst de amostra fornecido.")

        A_arr = np.asarray(A_vals, dtype=float)
        A_mean = float(A_arr.mean())
        A_sd = float(A_arr.std(ddof=1)) if len(A_arr) > 1 else 0.0
        A_cv = float(A_sd / A_mean * 100.0) if A_mean != 0 else None

        m = float(request.curve.m)
        b = float(request.curve.b)
        if m == 0:
            raise ValueError("Coeficiente m da curva não pode ser zero.")

        C_arr = (A_arr - b) / m
        C_mean = float(C_arr.mean())
        C_sd = float(C_arr.std(ddof=1)) if len(C_arr) > 1 else 0.0
        C_cv = float(C_sd / C_mean * 100.0) if C_mean != 0 else None

        results = dict(
            meta=dict(
                lambda_nm=request.target_wavelength,
                window_nm=request.window_nm,
                n_replicates=len(A_vals),
                pixel_to_nm_used=request.pixel_to_nm is not None,
                curve=dict(m=m, b=b, r2=request.curve.r2),
            ),
            absorbance=dict(
                replicates=A_vals,
                mean=A_mean,
                sd=A_sd,
                cv_percent=A_cv,
            ),
            concentration=dict(
                replicates=C_arr.tolist(),
                mean=C_mean,
                sd=C_sd,
                cv_percent=C_cv,
            ),
            qa=dict(
                notes="ok",
                absorbance_range_ok=(0.1 <= A_mean <= 1.0),
                high_cv_warn=(A_cv is not None and A_cv > 5.0),
            ),
        )

        return results

    def run_quantitative(self, request: QuantAnalyzeRequest) -> QuantAnalyzeResponse:
        try:
            results = self._analyze_quantitative(request)
            return QuantAnalyzeResponse(status="success", results=results)
        except Exception as e:
            return QuantAnalyzeResponse(status="error", error=str(e))

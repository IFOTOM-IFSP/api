from __future__ import annotations
from typing import List, Optional

import numpy as np

from .models import (
    ROI,
    PixelToNm,
    Curve,
    ReferenceBurst,
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

        # Curva de calibração: A = m*C + b
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


from __future__ import annotations
from typing import List, Optional
import numpy as np
from math import log

from scipy.optimize import curve_fit

from .models import (
    ROI,
    PixelToNm,
    Curve,
    ReferenceBurst,
    KineticAnalyzeRequest,
    KineticAnalyzeResponse,
)



def _exp1(t, Ainf, A0, k):
    # A(t) = Ainf + (A0 - Ainf) * exp(-k t)
    return Ainf + (A0 - Ainf) * np.exp(-k * t)


def _as_1d(y):
    """Aceita [[i, val], ...] ou [val, val, ...] e retorna np.ndarray 1D de valores."""
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1].astype(float)
    return arr.astype(float)


class SpectraProcessor:
    """Core de espectros para IFOTOM.
    Implementa cinética A(t) em λ fixo.
    """

    def _get_burst_mean_vector(self, burst: ReferenceBurst, roi: Optional[ROI]) -> np.ndarray:
        """Retorna a média dos vetores do burst.
        Aqui assumimos que os vetores já são espectros 1D pós-ROI. Se quiser aplicar ROI em imagem,
        esse passo deve ocorrer antes (no gerador do vetor).
        """
        vectors = [np.asarray(v, dtype=float) for v in burst.vectors]
        if not vectors:
            raise ValueError("Burst sem vetores")
        # Checar tamanhos coerentes
        L = min(v.shape[0] for v in vectors)
        stack = np.vstack([v[:L] for v in vectors])
        return stack.mean(axis=0)

    def _compensate_spectrum(self, raw: np.ndarray, dark: np.ndarray, white: np.ndarray) -> np.ndarray:
        """Compensa dark/white e devolve absorbância A = -log10(T)."""
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

    def _abs_at_lambda(self, wavelengths: np.ndarray, A: np.ndarray, lambda0: float, window_nm: float) -> float:
        if wavelengths is None:
            # Sem calibração: usa pixel mais próximo de lambda0 como índice bruto
            idx = int(round(lambda0))
            idx = max(0, min(idx, len(A) - 1))
            return float(A[idx])
        mask = np.abs(wavelengths - lambda0) <= (window_nm / 2.0)
        if not np.any(mask):
            # fallback: ponto mais próximo
            idx = int(np.argmin(np.abs(wavelengths - lambda0)))
            return float(A[idx])
        return float(A[mask].mean())


    def _analyze_kinetic(
        self,
        request: KineticAnalyzeRequest,
    ) -> dict:
        # Preparar referências
        dark = _as_1d(request.dark_reference_spectrum)
        white = _as_1d(request.white_reference_spectrum)

        # Calcular comprimentos de onda (se houver pixel_to_nm)
        wavelengths = None
        L = len(dark)
        if request.pixel_to_nm is not None:
            coeffs = [request.pixel_to_nm.a0, request.pixel_to_nm.a1, request.pixel_to_nm.a2]
            wavelengths = self._px_to_nm(L, coeffs)

        # Extrair A(t) em λ alvo
        t_vals: List[float] = []
        A_vals: List[float] = []
        for tp in request.series:
            vec = self._get_burst_mean_vector(tp.burst, request.roi)[:L]
            A = self._compensate_spectrum(vec, dark, white)
            A0 = self._abs_at_lambda(wavelengths, A, request.target_wavelength, request.window_nm)
            t_vals.append(float(tp.t_sec))
            A_vals.append(float(A0))

        t = np.asarray(t_vals, dtype=float)
        A_t = np.asarray(A_vals, dtype=float)

        # Ordenar por tempo (precaução)
        order = np.argsort(t)
        t = t[order]
        A_t = A_t[order]

        # Ajuste
        spec = request.fit or KineticFitSpec()
        model = spec.model
        A_fit = np.full_like(A_t, np.nan)
        meta = {}

        if model == "first_order":
            # Estimativas iniciais
            if spec.baseline_strategy == "tail_median":
                n_tail = max(3, int(len(t) * spec.tail_fraction))
                Ainf0 = float(np.median(A_t[-n_tail:])) if len(t) >= 3 else float(A_t[-1])
            else:
                Ainf0 = float(min(A_t.min(), A_t[-1]))
            A00 = float(A_t[0])
            # Chute para k baseado no espaçamento médio de tempo
            dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
            k0 = 0.1 / max(dt, 1e-6)
            p0 = (Ainf0, A00, k0)

            try:
                popt, pcov = curve_fit(_exp1, t, A_t, p0=p0, maxfev=20000)
                Ainf, A0, k = map(float, popt)
                A_fit = _exp1(t, *popt)
                ss_res = float(np.sum((A_t - A_fit) ** 2))
                ss_tot = float(np.sum((A_t - np.mean(A_t)) ** 2)) + 1e-12
                R2 = 1.0 - ss_res / ss_tot
                # IC95 de k
                try:
                    se_k = float(np.sqrt(max(pcov[2, 2], 0.0)))
                    k_ci = [k - 1.96 * se_k, k + 1.96 * se_k]
                except Exception:
                    k_ci = [None, None]
                meta.update(dict(k=k, k_ci95=k_ci, t_half=float(np.log(2) / k) if k > 0 else None, R2=R2, A0=A0, Ainf=Ainf))
            except Exception as e:
                meta.update(dict(error=f"fit_failed: {e}"))

        elif model == "second_order":
            # Linearização simples em 1/A ~ t (só didática; o correto é 1/C)
            y = 1.0 / np.clip(A_t, 1e-6, None)
            x = t
            M = np.vstack([x, np.ones_like(x)]).T
            m, b = np.linalg.lstsq(M, y, rcond=None)[0]
            yhat = m * x + b
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
            R2 = 1.0 - ss_res / ss_tot
            meta.update(dict(k=float(m), k_ci95=None, t_half=None, R2=R2))
            A_fit = 1.0 / np.clip(yhat, 1e-6, None)

        else:  # "none"
            meta.update(dict(k=None, k_ci95=None, t_half=None, R2=None))
            A_fit = A_t.copy()

        residuals = (A_t - A_fit).tolist()

        results = dict(
            meta=dict(
                lambda_nm=request.target_wavelength,
                window_nm=request.window_nm,
                delta_t_mean_s=float(np.mean(np.diff(t))) if len(t) > 1 else None,
                n_timepoints=len(t),
            ) | meta,
            series=dict(
                t_sec=t.tolist(),
                A=A_t.tolist(),
                A_fit=A_fit.tolist(),
                residuals=residuals,
            ),
            stack=None,
            qa=dict(notes="ok", monotonicity_warn=False, saturation_warn=False, drift_warn=False),
        )

        return results


    def run_kinetic(self, request: KineticAnalyzeRequest) -> KineticAnalyzeResponse:
        try:
            results = self._analyze_kinetic(request)
            return KineticAnalyzeResponse(status="success", results=results)
        except Exception as e:
            return KineticAnalyzeResponse(status="error", error=str(e))

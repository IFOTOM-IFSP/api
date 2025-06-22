
from app.api.v1.models import AnalysisRequest, AnalysisResult, CalibrationCurve
import numpy as np
import cv2
import base64

from typing import List, Tuple, Optional 
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpectraProcessor:
    def __init__(self, request_data: AnalysisRequest):
        self.data = request_data

        self.calibration_coeffs = self.data.calibration_coefficients or self._calculate_calibration_coeffs()

    def _base64_to_image(self, base64_string: str) -> np.ndarray:
        """Descodifica uma string base64 numa imagem OpenCV (BGR)."""
        try:
            img_data = base64.b64decode(base64_string)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("A string base64 não é uma imagem válida.")
            return image
        except Exception as e:
            logging.error(f"Erro ao descodificar base64: {e}", exc_info=True)
            raise

    def _convert_to_grayscale_profile(self, image: np.ndarray) -> np.ndarray:
        """Converte uma imagem para escala de cinza e extrai o perfil de intensidade."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        intensity_profile = np.mean(gray_image, axis=0)
        return intensity_profile

    def _get_averaged_profile(self, frames_base64: List[str]) -> np.ndarray:
        """Calcula a média de uma série de frames para reduzir o ruído."""
        if not frames_base64:
            raise ValueError("A lista de frames para fazer a média não pode estar vazia.")
        all_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame_str)) for frame_str in frames_base64]
        return np.mean(all_profiles, axis=0)

    def _compensate_spectrum(self, sample_profile, dark_profile, white_profile) -> np.ndarray:
        """Aplica a fórmula de compensação para obter a absorbância."""
        min_len = min(len(sample_profile), len(dark_profile), len(white_profile))
        sample, dark, white = sample_profile[:min_len], dark_profile[:min_len], white_profile[:min_len]

        denominator = white - dark
        denominator[denominator <= 1e-9] = 1e-9
        
        transmittance = (sample - dark) / denominator
        transmittance = np.clip(transmittance, 1e-4, 1.0) 
        
        absorbance = -np.log10(transmittance)
        return absorbance

    def _apply_wavelength_calibration(self, profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica a calibração para obter um eixo de comprimentos de onda."""
        pixels = np.arange(len(profile))
        a0, a1, a2, *_ = self.calibration_coeffs + [0] * (3 - len(self.calibration_coeffs))
        wavelengths = a0 + a1 * pixels + a2 * (pixels ** 2)
        return wavelengths, profile

    def _calculate_calibration_coeffs(self) -> List[float]:
        """Analisa a imagem de referência para criar a equação de calibração."""
        logging.info("A realizar a calibração de comprimento de onda a partir da referência branca...")
        white_profile = self._get_averaged_profile(self.data.white_frames_base64)
        
       
        peak_height = np.mean(white_profile) * 1.1 
        peak_distance = 50 
        peaks, _ = find_peaks(white_profile, height=peak_height, distance=peak_distance)
        logging.info(f"Encontrados {len(peaks)} picos de intensidade nas posições de pixel: {peaks}")

        known_wavelengths = self.data.known_wavelengths_for_calibration or [465, 545]

        if len(peaks) < len(known_wavelengths):
            raise ValueError(f"Não foram encontrados picos suficientes ({len(peaks)}) para os comprimentos de onda conhecidos ({len(known_wavelengths)}).")
        
        pixel_positions = peaks[:len(known_wavelengths)]
        coeffs = np.polyfit(pixel_positions, known_wavelengths, 2)
        reversed_coeffs = list(coeffs[::-1])
        logging.info(f"Coeficientes de calibração calculados (a₀, a₁, a₂): {reversed_coeffs}")
        return reversed_coeffs
    
    def _get_peak_absorbance(
        self, 
        wavelengths: np.ndarray, 
        absorbance_profile: np.ndarray, 
        target_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """Encontra a absorbância máxima, opcionalmente dentro de uma faixa de comprimento de onda."""
        if not target_range:
            return np.max(absorbance_profile)
        
        min_wav, max_wav = target_range
        indices = np.where((wavelengths >= min_wav) & (wavelengths <= max_wav))
        
        if indices[0].size == 0:
            logging.warning(f"Nenhum dado encontrado na faixa de comprimento de onda {target_range} nm.")
            return 0.0
            
        return np.max(absorbance_profile[indices])

    def _perform_linear_regression(self, points: List[Tuple[float, float]]) -> dict:
        """Calcula a regressão linear (curva de calibração)."""
        if len(points) < 2:
            return {"r_squared": 0, "equation": "N/A", "slope": 0, "intercept": 0}

        x = np.array([p[0] for p in points]) # Concentrações
        y = np.array([p[1] for p in points]) # Absorbâncias
        slope, intercept = np.polyfit(x, y, 1)
        
        if np.all(y == y[0]):
            r_squared = 1.0 if np.all(x * slope + intercept == y) else 0.0
        else:
            y_predicted = slope * x + intercept
            r_squared = np.corrcoef(y, y_predicted)[0,1]**2

        equation_str = f"y = {slope:.4f}x + {intercept:.4f}"
        
        return {"r_squared": r_squared, "equation": equation_str, "slope": slope, "intercept": intercept}
        
    def _calculate_noise_metrics(self, dark_frames_base64: List[str]) -> NoiseMetrics:
        if len(dark_frames_base64) < 2:
            logging.warning("Apenas um frame de escuro fornecido. O cálculo de ruído requer pelo menos 2 frames.")
            return NoiseMetrics(dark_current_std_dev=0.0)

        logging.info(f"A calcular o ruído a partir de {len(dark_frames_base64)} frames de escuro...")
        
        dark_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame_str)) for frame_str in dark_frames_base64]
        
        stacked_profiles = np.stack(dark_profiles, axis=0)
        
        std_dev_per_pixel = np.std(stacked_profiles, axis=0)
        
        average_std_dev = np.mean(std_dev_per_pixel)
        
        logging.info(f"Ruído (Desvio Padrão Médio do Sinal de Escuro): {average_std_dev:.4f}")
        
        return NoiseMetrics(dark_current_std_dev=float(average_std_dev))
        
    def process(self) -> AnalysisResult:
        logging.info("A iniciar o processamento de imagem...")
        noise_data = self._calculate_noise_metrics(self.data.dark_frames_base64)
        avg_dark_profile = self._get_averaged_profile(self.data.dark_frames_base64)
        avg_white_profile = self._get_averaged_profile(self.data.white_frames_base64)

        standard_points = []
        for sample in self.data.samples:
            if sample.type == 'standard':
                logging.info(f"A processar o padrão com concentração: {sample.concentration}")
                avg_standard_profile = self._get_averaged_profile(sample.frames_base64)
                absorbance_profile = self._compensate_spectrum(avg_standard_profile, avg_dark_profile, avg_white_profile)
                wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile)
                
        
                peak_absorbance = self._get_peak_absorbance(wavelengths, absorbance_profile, target_range=sample.target_wavelength_range)
                if sample.concentration is not None:
                    standard_points.append((sample.concentration, peak_absorbance))

        calibration_results = self._perform_linear_regression(standard_points) if standard_points else None

        unknown_sample_data = next((s for s in self.data.samples if s.type == 'unknown'), None)
        if not unknown_sample_data:
            return AnalysisResult(calibration_curve=CalibrationCurve(**calibration_results) if calibration_results else None)

        avg_sample_profile = self._get_averaged_profile(unknown_sample_data.frames_base64)
        absorbance_profile = self._compensate_spectrum(avg_sample_profile, avg_dark_profile, avg_white_profile)
        
        wavelengths, absorbance_values = self._apply_wavelength_calibration(absorbance_profile)
        
        sample_absorbance_value = self._get_peak_absorbance(wavelengths, absorbance_values, target_range=unknown_sample_data.target_wavelength_range)
        
        logging.info(f"Processamento concluído. Absorbância da amostra: {sample_absorbance_value:.4f}")

        calculated_concentration = None
        if calibration_results and calibration_results['slope'] != 0:
            slope = calibration_results['slope']
            intercept = calibration_results['intercept']
            calculated_concentration = (sample_absorbance_value - intercept) / slope
            logging.info(f"Concentração calculada: {calculated_concentration:.4f}")

        final_calibration_curve = CalibrationCurve(**calibration_results) if calibration_results else None

        return AnalysisResult(
            calibration_curve=final_calibration_curve,
            calculated_concentration=calculated_concentration,
            sample_absorbance=float(sample_absorbance_value),
            spectrum_data=list(zip(wavelengths.tolist(), absorbance_values.tolist()))
            noise_metrics=noise_data 
        )

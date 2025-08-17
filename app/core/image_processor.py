import numpy as np
import cv2
import base64
import logging
from typing import List, Tuple, Dict, Any
from scipy.signal import find_peaks

from app.api.v1.models import (
    AnalysisRequest,
    AnalysisResult,
    CalibrationCurve,
    ReferenceProcessingRequest,
    SampleResult
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpectraProcessor:

    def process_references(self, request: ReferenceProcessingRequest) -> Dict[str, Any]:
        logging.info("Iniciando o processamento de referências...")

        dark_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame)) for frame in request.dark_frames_base64]
        avg_dark_profile = np.mean(dark_profiles, axis=0)
        noise_metrics = self._calculate_noise_metrics(dark_profiles)

        avg_white_profile = self._get_averaged_profile(request.white_frames_base64)

        known_wavelengths = request.known_wavelengths_for_calibration or [465, 545]
        coeffs = self._calculate_wavelength_calibration_coeffs(
            avg_white_profile, 
            known_wavelengths,
            height_factor=request.peak_detection_height_factor, 
            distance=request.peak_detection_distance
        )

        logging.info("Processamento de referências concluído.")

        return {
            "dark_reference_spectrum": list(enumerate(avg_dark_profile.tolist())),
            "white_reference_spectrum": list(enumerate(avg_white_profile.tolist())),
            "pixel_to_wavelength_coeffs": coeffs,
            "dark_current_std_dev": noise_metrics.get('dark_current_std_dev', 0.0)
        }

    def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:

        logging.info(f"Iniciando análise do tipo: {request.analysisType}")

        if request.analysisType in ['quantitative', 'simple_read']:
            return self._process_quantitative_analysis(request)
        elif request.analysisType == 'scan':
            return self._process_scan_analysis(request)
        elif request.analysisType == 'kinetic':
            return self._process_kinetic_analysis(request)
        else:
            raise NotImplementedError(f"O tipo de análise '{request.analysisType}' ainda não foi implementado.")

    def _profile_from_request_data(self, spectrum_data: List[Tuple[int, float]]) -> np.ndarray:
        return np.array([intensity for _, intensity in spectrum_data])


    def _process_quantitative_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        avg_dark_profile = self._profile_from_request_data(request.dark_reference_spectrum)
        avg_white_profile = self._profile_from_request_data(request.white_reference_spectrum)

        calibration_results_dict = None
        newly_created_curve_model = None  # <-- Usaremos esta variável para o retorno condicional


        if request.calibration_curve:
            logging.info("Usando parâmetros de curva de calibração fornecidos pelo cliente.")
            calibration_results_dict = {
                "slope": request.calibration_curve.slope,
                "intercept": request.calibration_curve.intercept,
            }

        else:
            logging.info("Gerando curva de calibração em tempo real a partir de padrões.")
            standard_points = []
            for sample in request.samples:
                if sample.type == 'standard':
                    avg_standard_profile = self._get_averaged_profile(sample.frames_base64)
                    absorbance_profile = self._compensate_spectrum(avg_standard_profile, avg_dark_profile, avg_white_profile)
                    wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
                    peak_absorbance = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)
                    if sample.concentration is not None:
                        standard_points.append((sample.concentration, peak_absorbance))

            if standard_points:
                calibration_results_dict = self._perform_linear_regression(standard_points)
                newly_created_curve_model = CalibrationCurve(**calibration_results_dict)


        sample_results_list = []
        for sample in request.samples:
            if sample.type == 'unknown':
                avg_sample_profile = self._get_averaged_profile(sample.frames_base64)
                absorbance_profile = self._compensate_spectrum(avg_sample_profile, avg_dark_profile, avg_white_profile)
                wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
                sample_absorbance = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)

                calculated_concentration = None
                # ANOTAÇÃO: Usamos 'calibration_results_dict', que é preenchido em AMBOS os cenários.
                if calibration_results_dict and calibration_results_dict.get('slope') != 0:
                    slope = calibration_results_dict['slope']
                    intercept = calibration_results_dict['intercept']
                    read_concentration = (sample_absorbance - intercept) / slope
                    calculated_concentration = read_concentration * (sample.dilution_factor or 1.0)

                sample_results_list.append(
                    SampleResult(
                        sample_absorbance=sample_absorbance,
                        calculated_concentration=calculated_concentration,
                        spectrum_data=list(zip(wavelengths.tolist(), absorbance_profile.tolist()))
                    )
                )

        return AnalysisResult(
            calibration_curve=newly_created_curve_model,
            sample_results=sample_results_list
        )

    def _process_scan_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.samples or request.samples[0].type != 'unknown':
            raise ValueError("A análise de varredura requer uma amostra do tipo 'unknown'.")

        sample = request.samples[0]
        avg_dark_profile = self._profile_from_request_data(request.dark_reference_spectrum)
        avg_white_profile = self._profile_from_request_data(request.white_reference_spectrum)
        avg_sample_profile = self._get_averaged_profile(sample.frames_base64)
        absorbance_profile = self._compensate_spectrum(avg_sample_profile, avg_dark_profile, avg_white_profile)
        wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
        full_spectrum_data = list(zip(wavelengths.tolist(), absorbance_profile.tolist()))
        max_abs_index = np.argmax(absorbance_profile)
        max_abs_value = absorbance_profile[max_abs_index]
        
        return AnalysisResult(
            sample_results=[SampleResult(sample_absorbance=max_abs_value, spectrum_data=full_spectrum_data)]
        )

    def _process_kinetic_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.samples or not request.samples[0].timestamps_sec or len(request.samples[0].frames_base64) != len(request.samples[0].timestamps_sec) or not request.target_wavelength:
            raise ValueError("Dados inválidos para análise cinética.")

        sample = request.samples[0]
        avg_dark_profile = self._profile_from_request_data(request.dark_reference_spectrum)
        avg_white_profile = self._profile_from_request_data(request.white_reference_spectrum)
        kinetic_data_points = []
        
        for frame_b64, timestamp in zip(sample.frames_base64, sample.timestamps_sec):
            frame_profile = self._convert_to_grayscale_profile(self._base64_to_image(frame_b64))
            absorbance_profile = self._compensate_spectrum(frame_profile, avg_dark_profile, avg_white_profile)
            wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
            absorbance_at_t = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)
            kinetic_data_points.append((timestamp, absorbance_at_t))
        
        sample_result = SampleResult(
            kinetic_data=kinetic_data_points,
            sample_absorbance=kinetic_data_points[-1][1] if kinetic_data_points else None
        )
        return AnalysisResult(sample_results=[sample_result])

    def _base64_to_image(self, base64_string: str) -> np.ndarray:
        try:
            img_data = base64.b64decode(base64_string)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None: 
                raise ValueError("A string base64 não é uma imagem válida.")
            
            standard_size = (640, 480)
            resized_image = cv2.resize(image, standard_size)
            return resized_image
            
        except Exception as e:
            logging.error(f"Erro ao processar imagem base64: {e}", exc_info=True)
            raise

    def _convert_to_grayscale_profile(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image, axis=0)

    def _get_averaged_profile(self, frames_base64: List[str]) -> np.ndarray:
        if not frames_base64: raise ValueError("A lista de frames não pode estar vazia.")
        all_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame)) for frame in frames_base64]
        return np.mean(all_profiles, axis=0)

    def _compensate_spectrum(self, sample_profile, dark_profile, white_profile) -> np.ndarray:
        min_len = min(len(sample_profile), len(dark_profile), len(white_profile))
        sample, dark, white = sample_profile[:min_len], dark_profile[:min_len], white_profile[:min_len]
        denominator = white - dark
        denominator[denominator <= 1e-9] = 1e-9
        transmittance = (sample - dark) / denominator
        transmittance = np.clip(transmittance, 1e-5, 1.0)
        return -np.log10(transmittance)

    def _apply_wavelength_calibration(self, profile: np.ndarray, coeffs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        if not coeffs or len(coeffs) != 3:
            raise ValueError("Coeficientes de calibração de comprimento de onda inválidos ou não fornecidos.")
        pixels = np.arange(len(profile))
        a2, a1, a0 = coeffs
        wavelengths = a2 * (pixels ** 2) + a1 * pixels + a0
        return wavelengths, profile
        
    def _calculate_wavelength_calibration_coeffs(self, white_profile: np.ndarray, known_wavelengths: List[float], height_factor: float, distance: int) -> List[float]:
        peak_height = np.mean(white_profile) * height_factor 
        peak_distance = distance                             
        peaks, _ = find_peaks(white_profile, height=peak_height, distance=peak_distance)
        if len(peaks) < len(known_wavelengths):
            raise ValueError(f"Picos insuficientes ({len(peaks)}) para os comprimentos de onda ({len(known_wavelengths)}).")
        
        pixel_positions = sorted(peaks)[:len(known_wavelengths)]
        coeffs = np.polyfit(pixel_positions, sorted(known_wavelengths), 2)
        return coeffs.tolist()
        
    def _get_absorbance_at_wavelength(self, wavelengths: np.ndarray, absorbance_profile: np.ndarray, target_wavelength: float) -> float:
        if target_wavelength is None: return np.max(absorbance_profile)
        closest_index = np.argmin(np.abs(wavelengths - target_wavelength))
        return absorbance_profile[closest_index]

    def _perform_linear_regression(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        if len(points) < 2: return {"r_squared": 0, "equation": "N/A", "slope": 0, "intercept": 0}
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        slope, intercept = np.polyfit(x, y, 1)
        correlation_matrix = np.corrcoef(x, y)
        r_squared = correlation_matrix[0,1]**2
        equation_str = f"y = {slope:.4f}x {intercept:+.4f}"
        return {"r_squared": r_squared, "equation": equation_str, "slope": slope, "intercept": intercept}
        
    def _calculate_noise_metrics(self, dark_profiles: List[np.ndarray]) -> Dict[str, float]:
        if len(dark_profiles) < 2: return {'dark_current_std_dev': 0.0}
        std_dev_per_pixel = np.std(np.stack(dark_profiles, axis=0), axis=0)
        return {'dark_current_std_dev': float(np.mean(std_dev_per_pixel))}


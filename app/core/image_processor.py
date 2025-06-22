
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
    """
    Classe responsável por todo o processamento de imagens e cálculos espectrais.
    Esta classe foi refatorada para ser 'stateless', ou seja, não armazena o estado de um pedido.
    Cada método público executa uma tarefa específica com os dados que recebe.
    """


    def process_references(self, request: ReferenceProcessingRequest) -> Dict[str, Any]:
        """
        Processa as imagens de referência (escuro e branco) para gerar espectros médios.
        Esta função corresponde ao endpoint /process-references.
        """
        logging.info("Iniciando o processamento de referências...")

        dark_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame)) for frame in request.dark_frames_base64]
        avg_dark_profile = np.mean(dark_profiles, axis=0)
        noise_metrics = self._calculate_noise_metrics(dark_profiles)

        avg_white_profile = self._get_averaged_profile(request.white_frames_base64)
        
        coeffs = None
        if request.known_wavelengths_for_calibration:
            coeffs = self._calculate_wavelength_calibration_coeffs(
                avg_white_profile, 
                request.known_wavelengths_for_calibration
            )

        logging.info("Processamento de referências concluído.")
        
        return {
            "dark_reference_spectrum": list(enumerate(avg_dark_profile.tolist())),
            "white_reference_spectrum": list(enumerate(avg_white_profile.tolist())),
            "pixel_to_wavelength_coeffs": coeffs,
            "dark_current_std_dev": noise_metrics.get('dark_current_std_dev', 0.0)
        }



    def _process_quantitative_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Processa uma análise quantitativa ou uma leitura simples."""
        
        avg_dark_profile = np.array([intensity for _, intensity in request.dark_reference_spectrum])
        avg_white_profile = np.array([intensity for _, intensity in request.white_reference_spectrum])
        
        standard_points = []
        for sample in request.samples:
            if sample.type == 'standard':
                logging.info(f"Processando padrão com concentração: {sample.concentration}")
                avg_standard_profile = self._get_averaged_profile(sample.frames_base64)
                
                absorbance_profile = self._compensate_spectrum(avg_standard_profile, avg_dark_profile, avg_white_profile)
                wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
                
                peak_absorbance = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)
                
                if sample.concentration is not None:
                    standard_points.append((sample.concentration, peak_absorbance))

        calibration_results = self._perform_linear_regression(standard_points) if standard_points else None
        
        sample_results_list = []
        for sample in request.samples:
            if sample.type == 'unknown':
                logging.info(f"Processando amostra desconhecida. Fator de diluição: {sample.dilution_factor}")
                avg_sample_profile = self._get_averaged_profile(sample.frames_base64)
                
                absorbance_profile = self._compensate_spectrum(avg_sample_profile, avg_dark_profile, avg_white_profile)
                wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
                
                sample_absorbance = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)
                
                calculated_concentration = None
                if calibration_results and calibration_results['slope'] != 0:
                    slope = calibration_results['slope']
                    intercept = calibration_results['intercept']
                    
                    read_concentration = (sample_absorbance - intercept) / slope
                    
                    calculated_concentration = read_concentration * sample.dilution_factor

                sample_results_list.append(
                    SampleResult(
                        sample_absorbance=sample_absorbance,
                        calculated_concentration=calculated_concentration,
                        spectrum_data=list(zip(wavelengths.tolist(), absorbance_profile.tolist()))
                    )
                )
        
        return AnalysisResult(
            calibration_curve=CalibrationCurve(**calibration_results) if calibration_results else None,
            sample_results=sample_results_list
        )

    def _process_scan_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Processa uma análise de varredura de comprimento de onda."""
        if not request.samples or request.samples[0].type != 'unknown':
            raise ValueError("A análise de varredura requer uma amostra do tipo 'unknown'.")

        sample = request.samples[0]
        logging.info("Processando análise de varredura (scan).")

        avg_dark_profile = np.array([intensity for _, intensity in request.dark_reference_spectrum])
        avg_white_profile = np.array([intensity for _, intensity in request.white_reference_spectrum])
        
        avg_sample_profile = self._get_averaged_profile(sample.frames_base64)
        absorbance_profile = self._compensate_spectrum(avg_sample_profile, avg_dark_profile, avg_white_profile)
        wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)

        full_spectrum_data = list(zip(wavelengths.tolist(), absorbance_profile.tolist()))

        max_abs_index = np.argmax(absorbance_profile)
        max_abs_value = absorbance_profile[max_abs_index]
        lambda_max = wavelengths[max_abs_index]

        logging.info(f"Pico de absorbância encontrado: {max_abs_value:.4f} a {lambda_max:.2f} nm")

        return AnalysisResult(
            sample_results=[SampleResult(sample_absorbance=max_abs_value, spectrum_data=full_spectrum_data)]
        )

     def run_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Executa a análise principal, direcionando para o método correto com base no tipo de análise.
        """
        logging.info(f"Iniciando análise do tipo: {request.analysisType}")
        
        if request.analysisType == 'quantitative' or request.analysisType == 'simple_read':
            return self._process_quantitative_analysis(request)
        elif request.analysisType == 'scan':
            return self._process_scan_analysis(request)
        # --> MUDANÇA: Adicionado o direcionamento para a análise cinética.
        elif request.analysisType == 'kinetic':
            return self._process_kinetic_analysis(request)
        else:
            raise NotImplementedError(f"O tipo de análise '{request.analysisType}' ainda não foi implementado.")

    # NOVO MÉTODO COMPLETO PARA ANÁLISE CINÉTICA
    def _process_kinetic_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """Processa uma análise cinética medindo a absorbância ao longo do tempo."""
        # Validação da requisição
        if not request.samples:
            raise ValueError("Análise cinética requer uma amostra com frames.")
        if not request.samples[0].timestamps_sec:
            raise ValueError("Análise cinética requer timestamps para cada frame.")
        if len(request.samples[0].frames_base64) != len(request.samples[0].timestamps_sec):
            raise ValueError("O número de frames e de timestamps deve ser o mesmo.")
        if not request.target_wavelength:
            raise ValueError("Análise cinética requer um 'target_wavelength'.")

        sample = request.samples[0]
        logging.info(f"Processando análise cinética em {request.target_wavelength} nm...")

        # Extrai os espectros de referência
        avg_dark_profile = np.array([intensity for _, intensity in request.dark_reference_spectrum])
        avg_white_profile = np.array([intensity for _, intensity in request.white_reference_spectrum])

        kinetic_data_points = []
        
        # Itera sobre cada frame e seu respectivo timestamp
        for frame_b64, timestamp in zip(sample.frames_base64, sample.timestamps_sec):
            # Processamento de imagem para obter a absorbância
            frame_image = self._base64_to_image(frame_b64)
            frame_profile = self._convert_to_grayscale_profile(frame_image)
            
            absorbance_profile = self._compensate_spectrum(frame_profile, avg_dark_profile, avg_white_profile)
            wavelengths, _ = self._apply_wavelength_calibration(absorbance_profile, request.pixel_to_wavelength_coeffs)
            
            # Obtém a absorbância no comprimento de onda alvo para este ponto no tempo
            absorbance_at_t = self._get_absorbance_at_wavelength(wavelengths, absorbance_profile, request.target_wavelength)
            
            kinetic_data_points.append((timestamp, absorbance_at_t))
        
        logging.info("Análise cinética concluída.")
        
        # Empacota os resultados no modelo de resposta
        sample_result = SampleResult(
            kinetic_data=kinetic_data_points,
            # Pode-se opcionalmente retornar a absorbância final como a principal
            sample_absorbance=kinetic_data_points[-1][1] if kinetic_data_points else None
        )
        
        return AnalysisResult(sample_results=[sample_result])

    def _base64_to_image(self, base64_string: str) -> np.ndarray:
        """Descodifica uma string base64 numa imagem OpenCV (BGR)."""
        try:
            img_data = base64.b64decode(base64_string)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None: raise ValueError("A string base64 não é uma imagem válida.")
            return image
        except Exception as e:
            logging.error(f"Erro ao descodificar base64: {e}", exc_info=True)
            raise

    def _convert_to_grayscale_profile(self, image: np.ndarray) -> np.ndarray:
        """Converte uma imagem para escala de cinza e extrai o perfil de intensidade médio."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image, axis=0)

    def _get_averaged_profile(self, frames_base64: List[str]) -> np.ndarray:
        """Calcula a média de perfis de uma série de frames para reduzir o ruído."""
        if not frames_base64: raise ValueError("A lista de frames não pode estar vazia.")
        all_profiles = [self._convert_to_grayscale_profile(self._base64_to_image(frame)) for frame in frames_base64]
        return np.mean(all_profiles, axis=0)

    def _compensate_spectrum(self, sample_profile, dark_profile, white_profile) -> np.ndarray:
        """Aplica a fórmula de compensação para obter a absorbância."""
        min_len = min(len(sample_profile), len(dark_profile), len(white_profile))
        sample, dark, white = sample_profile[:min_len], dark_profile[:min_len], white_profile[:min_len]

        denominator = white - dark
        denominator[denominator <= 1e-9] = 1e-9  # Evita divisão por zero ou valores muito pequenos
        
        transmittance = (sample - dark) / denominator
        transmittance = np.clip(transmittance, 1e-5, 1.0)  # Evita log de zero ou negativo e valores > 1.0
        
        absorbance = -np.log10(transmittance)
        return absorbance

    def _apply_wavelength_calibration(self, profile: np.ndarray, coeffs: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica a calibração polinomial para obter um eixo de comprimentos de onda."""
        pixels = np.arange(len(profile))
        # Garante que temos pelo menos 3 coeficientes (a0, a1, a2) para um polinômio de 2º grau
        a0, a1, a2, *_ = coeffs[::-1] + [0] * (3 - len(coeffs)) # Coeffs são armazenados [a2, a1, a0] por polyfit
        wavelengths = a2 * (pixels ** 2) + a1 * pixels + a0
        return wavelengths, profile
        
    def _calculate_wavelength_calibration_coeffs(self, white_profile: np.ndarray, known_wavelengths: List[float]) -> List[float]:
        """Analisa a imagem de referência para criar a equação de calibração de comprimento de onda."""
        logging.info("Realizando a calibração de comprimento de onda a partir da referência branca...")
        
        peak_height = np.mean(white_profile) * 1.1
        peak_distance = 50 # Distância mínima entre picos em pixels
        peaks, _ = find_peaks(white_profile, height=peak_height, distance=peak_distance)
        logging.info(f"Encontrados {len(peaks)} picos de intensidade nas posições de pixel: {peaks}")

        if len(peaks) < len(known_wavelengths):
            raise ValueError(f"Não foram encontrados picos suficientes ({len(peaks)}) para os comprimentos de onda conhecidos ({len(known_wavelengths)}).")
        
        pixel_positions = sorted(peaks)[:len(known_wavelengths)]
        # Assume que os known_wavelengths também estão ordenados
        coeffs = np.polyfit(pixel_positions, sorted(known_wavelengths), 2)
        
        logging.info(f"Coeficientes de calibração (a₂, a₁, a₀): {coeffs.tolist()}")
        return coeffs.tolist() # Retorna como [a2, a1, a0]
        
    def _get_absorbance_at_wavelength(self, wavelengths: np.ndarray, absorbance_profile: np.ndarray, target_wavelength: float) -> float:
        """Encontra a absorbância no comprimento de onda mais próximo do alvo."""
        if target_wavelength is None:
            # Se nenhum alvo for especificado, retorna o pico máximo como fallback
            return np.max(absorbance_profile)

        # Encontra o índice do comprimento de onda mais próximo do alvo
        closest_index = np.argmin(np.abs(wavelengths - target_wavelength))
        absorbance = absorbance_profile[closest_index]
        
        actual_wavelength = wavelengths[closest_index]
        logging.info(f"Absorbância no λ mais próximo de {target_wavelength}nm (que é {actual_wavelength:.2f}nm) é {absorbance:.4f}")
        return absorbance

    def _perform_linear_regression(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calcula a regressão linear para a curva de calibração."""
        if len(points) < 2:
            return {"r_squared": 0, "equation": "N/A", "slope": 0, "intercept": 0}

        x = np.array([p[0] for p in points])  # Concentrações
        y = np.array([p[1] for p in points])  # Absorbâncias
        slope, intercept = np.polyfit(x, y, 1)
        
        # Cálculo robusto do R²
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        
        equation_str = f"y = {slope:.4f}x + {intercept:+.4f}" # Usando + para mostrar o sinal do intercepto
        
        return {"r_squared": r_squared, "equation": equation_str, "slope": slope, "intercept": intercept}
        
    def _calculate_noise_metrics(self, dark_profiles: List[np.ndarray]) -> Dict[str, float]:
        """Calcula o ruído (desvio padrão) a partir de uma lista de perfis de escuro já processados."""
        if len(dark_profiles) < 2:
            logging.warning("Apenas um frame de escuro fornecido. O cálculo de ruído requer pelo menos 2.")
            return {'dark_current_std_dev': 0.0}

        stacked_profiles = np.stack(dark_profiles, axis=0)
        std_dev_per_pixel = np.std(stacked_profiles, axis=0)
        average_std_dev = np.mean(std_dev_per_pixel)
        
        logging.info(f"Ruído (Desvio Padrão Médio do Sinal de Escuro): {average_std_dev:.4f}")
        return {'dark_current_std_dev': float(average_std_dev)}


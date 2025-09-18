# 📘 Spectrophotometry Analysis API

API baseada em **FastAPI** para processamento de dados de espectrofotometria.
Responsável por receber dados brutos (imagens ou vetores espectrais), aplicar correções de escuro/branco, calcular absorbâncias e executar análises quantitativas ou espectrais.

---

## 🔗 Base URL

```
http://<host>:10000/api/v1
```

---

## 🩺 Health Check

### `GET /health`

Verifica se a API está no ar.

**Response**:

```json
{ "status": "ok" }
```

---

## 📊 Processamento de Referências

### `POST /process-references`

Processa os bursts de **dark** (escuro) e **white** (branco) e retorna espectros agregados e métricas de ruído.

**Request Body** (`ReferenceProcessingRequest`):

```json
{
  "dark": {
    "vectors": [
      [100, 120, 130, 140],
      [98, 119, 132, 138]
    ]
  },
  "white": {
    "vectors": [
      [200, 220, 250, 260],
      [198, 222, 248, 258]
    ]
  },
  "roi": { "x": 100, "y": 200, "w": 1024, "h": 20 }
}
```

**Response** (`ReferenceProcessingResponse`):

```json
{
  "status": "success",
  "dark_reference_spectrum": [[0, 99.0], [1, 119.5], [2, 131.0], [3, 139.0]],
  "white_reference_spectrum": [[0, 199.0], [1, 221.0], [2, 249.0], [3, 259.0]],
  "dark_current_std_dev": 1.2,
  "pixel_to_wavelength": null
}
```

---

## 🔬 Análise Quantitativa / Espectral

### `POST /analyze`

Executa análise quantitativa, leitura simples, scan espectral ou análise cinética.

**Request Body** (`AnalysisRequest`):

```json
{
  "analysisType": "quantitative",
  "pixel_to_wavelength": {
    "coeffs": [350.0, 0.25],
    "rmse_nm": 0.5
  },
  "calibration_curve": {
    "slope": 0.012,
    "intercept": 0.001,
    "r_squared": 0.999
  },
  "target_wavelength": 500,
  "window_nm": 4,
  "roi": { "x": 100, "y": 200, "w": 1024, "h": 20 },
  "dark_reference_spectrum": [[0, 98.0], [1, 119.0], [2, 131.0], [3, 138.0]],
  "white_reference_spectrum": [[0, 199.0], [1, 220.0], [2, 250.0], [3, 260.0]],
  "samples": [
    {
      "kind": "unknown",
      "burst": {
        "vectors": [
          [150, 170, 190, 200],
          [152, 168, 191, 202]
        ]
      }
    }
  ]
}
```

**Response** (`AnalysisResponse`):

```json
{
  "status": "success",
  "results": {
    "calibration_curve": null,
    "sample_results": [
      {
        "sample_absorbance": 0.452,
        "calculated_concentration": 37.58,
        "spectrum_data": [
          [499.8, 0.450],
          [500.0, 0.452],
          [500.2, 0.454]
        ]
      }
    ],
    "qa": { "notes": "ok" }
  }
}
```

---

## 📑 Modelos de Dados

### ROI

```json
{ "x": 100, "y": 200, "w": 1024, "h": 20 }
```

### PixelToWavelength

```json
{
  "coeffs": [a0, a1, a2],
  "rmse_nm": 0.5,
  "dispersion_nm_per_px": 0.25
}
```

### CalibrationCurveInput

```json
{
  "slope": 0.012,
  "intercept": 0.001,
  "r_squared": 0.999,
  "lod": 0.01,
  "loq": 0.03
}
```

---

## 🚦 Códigos de Status

* **200 OK** → Requisição processada com sucesso.
* **400 Bad Request** → Erro de validação (ex.: payload incorreto).
* **500 Internal Server Error** → Erro inesperado no servidor.
* **501 Not Implemented** → Funcionalidade ainda não disponível (ex.: kinetic).

---

## 📌 Notas Importantes

* Para **produção**, prefira sempre enviar **vetores 1D** (já somados por coluna no dispositivo) em vez de imagens base64.
* Informe sempre o **ROI** utilizado no app para garantir consistência entre capturas.
* `target_wavelength` é obrigatório para `quantitative` e `simple_read`.
* `calibration_curve` deve ser incluída se você já tiver os coeficientes `m` e `b`.
* Se não houver curva, use o fluxo de construção (Cenário 3).

---

# 📌 Cenário 1 — Análise Direta (equipamento + curva já definidos)

Usuário já tem **perfil de equipamento** e **curva de calibração (m,b)**.

### Request

```json
POST /api/v1/analyze
{
  "analysisType": "quantitative",
  "pixel_to_wavelength": { "coeffs": [350, 0.25] },
  "calibration_curve": { "slope": 0.012, "intercept": 0.001 },
  "target_wavelength": 500,
  "window_nm": 4,
  "dark_reference_spectrum": [[0, 98], [1, 119], [2, 131], [3, 138]],
  "white_reference_spectrum": [[0, 199], [1, 220], [2, 250], [3, 260]],
  "samples": [
    {
      "kind": "unknown",
      "burst": {
        "vectors": [
          [150, 170, 190, 200],
          [152, 168, 191, 202]
        ]
      }
    }
  ]
}
```

### Response (exemplo esperado)

```json
{
  "status": "success",
  "results": {
    "sample_results": [
      {
        "sample_absorbance": 0.452,
        "calculated_concentration": 37.6,
        "spectrum_data": [[499.8, 0.45], [500.0, 0.452], [500.2, 0.454]]
      }
    ],
    "qa": { "notes": "ok" }
  }
}
```

---

# 📌 Cenário 2 — Análise com Calibração do Equipamento

Usuário já tem curva (m,b), mas precisa calibrar o **pixel → λ**.

### Passo 1 — Calibração do equipamento

```json
POST /api/v1/process-references
{
  "dark": {
    "vectors": [[98, 119, 131, 138], [99, 118, 130, 137]]
  },
  "white": {
    "vectors": [[199, 220, 250, 260], [198, 222, 248, 258]]
  }
}
```

→ Resposta incluirá espectros médios + `dark_current_std_dev`.

---

### Passo 2 — Análise com curva e perfil λ calibrado

```json
POST /api/v1/analyze
{
  "analysisType": "quantitative",
  "pixel_to_wavelength": { "coeffs": [350, 0.25] },
  "calibration_curve": { "slope": 0.012, "intercept": 0.001 },
  "target_wavelength": 500,
  "window_nm": 4,
  "dark_reference_spectrum": [[0, 98], [1, 119], [2, 131], [3, 138]],
  "white_reference_spectrum": [[0, 199], [1, 220], [2, 250], [3, 260]],
  "samples": [
    {
      "kind": "unknown",
      "burst": {
        "vectors": [
          [150, 170, 190, 200],
          [152, 168, 191, 202]
        ]
      }
    }
  ]
}
```

---

# 📌 Cenário 3 — Construção de Curva + Calibração

Aqui não há curva nem perfil λ. Usuário envia **padrões conhecidos** para construir a curva.

### Request

```json
POST /api/v1/analyze
{
  "analysisType": "quantitative",
  "pixel_to_wavelength": { "coeffs": [350, 0.25] },
  "target_wavelength": 500,
  "window_nm": 4,
  "dark_reference_spectrum": [[0, 98], [1, 119], [2, 131], [3, 138]],
  "white_reference_spectrum": [[0, 199], [1, 220], [2, 250], [3, 260]],
  "samples": [
    {
      "kind": "standard",
      "concentration": 10.0,
      "burst": { "vectors": [[120, 145, 160, 180]] }
    },
    {
      "kind": "standard",
      "concentration": 20.0,
      "burst": { "vectors": [[140, 165, 180, 200]] }
    },
    {
      "kind": "standard",
      "concentration": 40.0,
      "burst": { "vectors": [[160, 185, 210, 230]] }
    },
    {
      "kind": "unknown",
      "burst": { "vectors": [[150, 170, 190, 200]] }
    }
  ]
}
```

### Response (exemplo esperado)

```json
{
  "status": "success",
  "results": {
    "calibration_curve": {
      "slope": 0.0118,
      "intercept": 0.0009,
      "r_squared": 0.998,
      "equation": "A = 0.0118*C + 0.0009"
    },
    "sample_results": [
      {
        "sample_absorbance": 0.452,
        "calculated_concentration": 38.2,
        "spectrum_data": [[499.8, 0.45], [500.0, 0.452], [500.2, 0.454]]
      }
    ],
    "qa": { "notes": "ok" }
  }
}
```

---



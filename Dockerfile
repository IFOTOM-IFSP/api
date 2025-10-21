# Use uma imagem base leve
FROM python:3.12-slim

# Defina o diretório de trabalho
WORKDIR /app

# Instale dependências do sistema APENAS se realmente necessário
# Considere usar opencv-python-headless para evitar isso
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie e instale as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie TODO o código do diretório atual para o WORKDIR
# Isso vai copiar sua pasta "app" para dentro de /app, ficando /app/app/...
# Se seu main.py está em app/main.py, o CMD precisa refletir isso.
COPY . .

# Comando de início que usa a variável $PORT fornecida pelo Cloud Run
# Assumindo que seu main.py está em app/main.py, o alvo deve ser "app.main:app"
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:$PORT"]

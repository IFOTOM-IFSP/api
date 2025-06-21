# 1. Usar uma imagem base oficial e leve do Python
FROM python:3.9-slim

# 2. Definir o diretório de trabalho dentro do contentor
WORKDIR /code

# 3. [MUITO IMPORTANTE] Instalar as dependências de sistema do OpenCV
# Estas bibliotecas são necessárias para o OpenCV funcionar em servidores Linux.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 4. Copiar o ficheiro de dependências para o contentor
COPY ./requirements.txt /code/requirements.txt

# 5. Instalar as dependências Python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copiar todo o código da sua aplicação
COPY ./app /code/app

# 7. Expor a porta 10000, que é um padrão recomendado pelo Render
EXPOSE 10000

# 8. O comando para iniciar a API quando o contentor arrancar
# O host '0.0.0.0' é essencial para que a API seja acessível de fora.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]

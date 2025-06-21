iFOTOM Analysis API
Esta é a API de backend para o projeto iFOTOM, uma aplicação de espectrofotometria baseada em smartphone. A API é responsável por receber os dados de imagem capturados pela aplicação móvel, realizar o processamento de imagem complexo utilizando OpenCV e retornar os resultados analíticos, como absorbância e concentração.

O projeto foi construído com FastAPI, o que garante alta performance e uma documentação de API interativa gerada automaticamente.

Tecnologias Principais
Framework: FastAPI

Servidor: Uvicorn

Processamento de Imagem: OpenCV (Headless)

Cálculo Numérico: NumPy

Processamento de Sinal: SciPy (usada para encontrar picos nos espectros)

Validação de Dados: Pydantic (integrado com o FastAPI)

Configuração do Ambiente Local
Para executar este projeto localmente, você precisará ter o Python 3.9+ instalado. É altamente recomendado o uso de um ambiente virtual para gerir as dependências do projeto.

1. Crie e Ative um Ambiente Virtual
No terminal, na raiz do projeto (iFOTOM-api/), execute os seguintes comandos:

# Cria o ambiente virtual na pasta 'venv'
python3 -m venv venv

# Ativa o ambiente virtual (para Linux/macOS)
source venv/bin/activate

No Windows, o comando de ativação é venv\Scripts\activate.

2. Instale as Dependências
Com o ambiente virtual ativo, instale todas as bibliotecas necessárias a partir do ficheiro requirements.txt:

pip install -r requirements.txt

3. Execute o Servidor de Desenvolvimento
Agora você pode iniciar a API. O Uvicorn irá fornecer um servidor de alta performance com "live reload", que reinicia automaticamente sempre que você altera o código.

uvicorn app.main:app --reload

Se tudo estiver correto, você verá uma mensagem a indicar que o servidor está a correr em http://127.0.0.1:8000.

Documentação da API (Swagger UI)
Uma das maiores vantagens do FastAPI é a documentação automática. Com o seu servidor a correr, abra o seu navegador e aceda a um dos seguintes URLs:

http://127.0.0.1:8000/docs: Para a documentação interativa Swagger UI. Aqui, você pode ver todos os seus endpoints, os modelos de dados, e até mesmo testar a sua API diretamente do navegador.

http://127.0.0.1:8000/redoc: Para uma documentação alternativa e mais limpa, gerada pelo ReDoc.

Como Testar o Endpoint de Análise
Para testar o endpoint principal /api/v1/process-analysis localmente, você pode usar a ferramenta de linha de comando curl. Este comando simula o envio de dados pela aplicação móvel.

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "analysisType": "quantitative",
    "dark_frames_base64": ["ZGFya19mcmFtZV90ZXN0"],
    "white_frames_base64": ["d2hpdGVfZnJhbWVfdGVzdA=="],
    "samples": [
      {
        "type": "unknown",
        "frames_base64": [
          "c2FtcGxlXzFfdGVzdA=="
        ]
      }
    ]
  }' \
  http://127.0.0.1:8000/api/v1/process-analysis

Se o teste for bem-sucedido, você deverá receber uma resposta JSON com o estado de success e os resultados da análise simulada.

Deploy
Este projeto está configurado para ser feito o deploy facilmente como um serviço web utilizando Docker. A plataforma recomendada para a hospedagem é o Render.

O processo de deploy está detalhado no ficheiro Dockerfile, que lida com a instalação de todas as dependências do sistema (para o OpenCV) e do Python. Para fazer o deploy, basta ligar o seu repositório GitHub a um novo "Web Service" no Render e selecionar o ambiente Docker.

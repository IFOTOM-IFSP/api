# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import ...

app = FastAPI(title="IFOTOM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # ajuste p/ domínios do app em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router)

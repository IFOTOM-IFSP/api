from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import analysis

app = FastAPI(title="iFOTOM Analysis API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])

@app.get("/")
def read_root():
    """Endpoint raiz para uma verificação de saúde rápida."""
    return {"status": "iFOTOM Analysis API v1 is running"}

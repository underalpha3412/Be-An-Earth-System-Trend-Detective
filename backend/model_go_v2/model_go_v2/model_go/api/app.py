"""
FastAPI application instance for model_go.
Strictly headless, production-ready, zero-frontend microservice.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_go.config import PROFILE
from model_go.api.routes import router

app = FastAPI(
    title="model_go | Production Meteorological Forecasting Engine",
    description="Unified headless meteorological API merging Model Alpha (GBDT), Hybrid Physics-ML, and IMD 4D-Var Assimilation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external microservices and mobile backends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "model_go",
        "version": "1.0.0",
        "status": "ONLINE",
        "profile": PROFILE,
        "api_docs": "/docs",
        "health_check": "/v1/health"
    }


def start_server(host: str = "0.0.0.0", port: int = 8080):
    """Run server via uvicorn programmatically."""
    import uvicorn
    uvicorn.run("model_go.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()

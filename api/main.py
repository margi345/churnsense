from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, explain, retention
from src import __version__

app = FastAPI(
    title       = "ChurnSense API",
    description = "AI-powered customer churn prediction and retention intelligence",
    version     = __version__,
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(predict.router,   tags=["Prediction"])
app.include_router(explain.router,   tags=["Explainability"])
app.include_router(retention.router, tags=["Retention"])

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}

@app.get("/")
def root():
    return {
        "message": "ChurnSense API is running",
        "docs":    "/docs",
        "version": __version__
    }
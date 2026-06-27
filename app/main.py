from fastapi import FastAPI

app = FastAPI(
    title="Explainability-as-a-Service (EaaS)",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "EaaS is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
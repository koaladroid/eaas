from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.api.routes import router

app = FastAPI(
    title="Explainability-as-a-Service (EaaS)",
    version="1.0.0",
)

app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Invalid request body.",
            }
        },
    )


@app.get("/")
def root():
    return {
        "service": "Explainability-as-a-Service (EaaS)",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Explainability-as-a-Service",
        "version": app.version,
    }

from fastapi import FastAPI
from contextlib import asynccontextmanager

from core.ml_engine import ml_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_engine.load_models()
    yield
    ml_engine.upload_models()


app = FastAPI(lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
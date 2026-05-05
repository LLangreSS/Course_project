from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from api.verify import router as verify_router
from api.knowledge_base import router as knowledge_base_router
from core.ml_engine import ml_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_engine.load_models()
    yield
    ml_engine.upload_models()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify_router, prefix="/verify", tags=["Verification"])
app.include_router(knowledge_base_router, prefix="/knowledge_base", tags=["Knowledge Base"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
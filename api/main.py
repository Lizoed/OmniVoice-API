from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Add OmniVoice-master to python path so it can import omnivoice modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "OmniVoice-master"))

from .routers.openai_compatible import router as openai_router, get_backend
from .services.voice_manager import init_db
from .config import settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB (now async)
    print("Initializing database...")
    await init_db()
    
    # Pre-load ML model into VRAM
    print("Loading ML model into VRAM...")
    # This might take a while, but it's done once at startup.
    # We can run it in a thread to not block the async init completely, but lifespan runs sequentially anyway.
    import asyncio
    await asyncio.to_thread(get_backend)
    print("ML model loaded successfully!")
    
    yield

app = FastAPI(
    title="OmniVoice OpenAI-Compatible API",
    description="A FastAPI server wrapping OmniVoice with an OpenAI-compatible TTS API, mirroring Qwen3 architecture.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the OpenAI compatible router under /v1
app.include_router(openai_router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

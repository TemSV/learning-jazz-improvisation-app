from fastapi import FastAPI
from .routes import songs, recommendations, phrases

import uvicorn

app = FastAPI(
    title="Jazz Improvisation Learning API",
    description="API for analyzing songs, finding patterns, and recommending phrases.",
    version="0.1.0",
)


app.include_router(songs.router)
app.include_router(recommendations.router)
app.include_router(phrases.router)

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


print("API Application configured.")

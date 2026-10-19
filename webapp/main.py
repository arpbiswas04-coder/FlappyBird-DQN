import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from webapp.schemas import (
    HealthResponse,
    ModelStatusResponse,
    SessionControlRequest,
    SessionControlResponse,
    MetricsHistoryResponse,
)
from webapp.agent_service import AgentDemonstrationService

agent_service = AgentDemonstrationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register the active asyncio event loop for threadsafe broadcasts
    loop = asyncio.get_running_loop()
    agent_service.set_loop(loop)
    yield
    # Cleanup on application shutdown
    agent_service.stop()


app = FastAPI(
    title="Flappy Bird DQN Agent Dashboard API",
    description="Real-time web monitoring and control dashboard for the trained Flappy Bird DQN reinforcement learning agent.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/api/model-status", response_model=ModelStatusResponse)
def get_model_status():
    """Get current PyTorch DQN model, device, and environment status."""
    return agent_service.get_status()


@app.post("/api/session/start", response_model=SessionControlResponse)
def start_session(req: SessionControlRequest = SessionControlRequest()):
    """Start the demonstration session using the trained DQN policy."""
    result = agent_service.start(fps=req.fps or 30)
    return SessionControlResponse(**result)


@app.post("/api/session/pause", response_model=SessionControlResponse)
def pause_session():
    """Pause the demonstration session."""
    result = agent_service.pause()
    return SessionControlResponse(**result)


@app.post("/api/session/resume", response_model=SessionControlResponse)
def resume_session():
    """Resume the demonstration session."""
    result = agent_service.resume()
    return SessionControlResponse(**result)


@app.post("/api/session/reset", response_model=SessionControlResponse)
def reset_session():
    """Reset the current episode in the demonstration session."""
    result = agent_service.reset()
    return SessionControlResponse(**result)


@app.post("/api/session/stop", response_model=SessionControlResponse)
def stop_session():
    """Stop the demonstration session cleanly."""
    result = agent_service.stop()
    return SessionControlResponse(**result)


@app.get("/api/metrics/history", response_model=MetricsHistoryResponse)
def get_metrics_history():
    """Retrieve history of completed episodes and best reward."""
    return agent_service.get_metrics_history()


@app.websocket("/ws/session")
async def websocket_session_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live game frames, actions, Q-values, and episode metrics.
    Clients can also send control commands directly over WebSocket.
    """
    await websocket.accept()
    agent_service.register_connection(websocket)
    
    try:
        # Immediately send snapshot of current status upon connection
        status_data = agent_service.get_status().model_dump()
        history_data = agent_service.get_metrics_history().model_dump()
        
        await websocket.send_json({
            "type": "init_state",
            "model_status": status_data,
            "metrics": history_data,
            "current_episode": agent_service.current_episode,
            "current_episode_reward": agent_service.current_episode_reward,
            "current_score": agent_service.current_score,
            "current_step": agent_service.current_step,
            "last_frame": agent_service.last_frame_b64,
            "last_action": agent_service.last_action,
            "status": agent_service._status,
        })
        
        # Keep listening for incoming client messages or commands
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            if command == "start":
                fps = data.get("fps", 30)
                agent_service.start(fps=fps)
            elif command == "pause":
                agent_service.pause()
            elif command == "resume":
                agent_service.resume()
            elif command == "reset":
                agent_service.reset()
            elif command == "stop":
                agent_service.stop()
            elif command == "set_fps":
                fps = data.get("fps", 30)
                agent_service.target_fps = max(5, min(60, int(fps)))
            elif command == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket client error: {e}")
    finally:
        agent_service.unregister_connection(websocket)


import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")

if os.path.exists(FRONTEND_DIST):
    # Mount the /assets static bundle directory
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/", include_in_schema=False)
async def serve_root():
    """Serve frontend index.html if available, or informative JSON message."""
    if os.path.exists(FRONTEND_DIST):
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return JSONResponse(
        content={
            "message": "Flappy Bird DQN API running. Build frontend with 'npm run build' to view dashboard.",
            "health": "/api/health",
            "model_status": "/api/model-status",
            "docs": "/docs",
        }
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa_fallback(full_path: str):
    """
    Production SPA fallback route:
    - Never intercepts API routes (starts with /api/ or api)
    - Never intercepts WebSocket routes (starts with /ws/ or ws)
    - Serves static files directly if present in FRONTEND_DIST (e.g., favicon.ico, vite.svg)
    - Serves index.html for all client-side navigation routes
    """
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    if full_path == "ws" or full_path.startswith("ws/"):
        raise HTTPException(status_code=404, detail="WebSocket path not found")

    if os.path.exists(FRONTEND_DIST):
        target_file = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(target_file):
            return FileResponse(target_file)

        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Resource not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)

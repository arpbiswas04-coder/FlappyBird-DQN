# AI Learned Flappy Bird (DQN) & Full-Stack Web Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Render Ready](https://img.shields.io/badge/Render-Deploy%20Ready-black.svg)](https://render.com/)

A Deep Q-Network (DQN) reinforcement learning agent trained to play Flappy Bird, featuring a high-performance **FastAPI** backend, WebSocket streaming, and a responsive **Neo-Brutalist React web dashboard**.

- **Repository**: [https://github.com/arpbiswas04-coder/FlappyBird-DQN](https://github.com/arpbiswas04-coder/FlappyBird-DQN)
- **Original RL Baseline**: Based on [`Madhura-Kedar/Ai-learned-Flappy-Bird`](https://github.com/Madhura-Kedar/Ai-learned-Flappy-Bird) and [`flappy-bird-gymnasium`](https://github.com/Talendar/flappy-bird-gymnasium).

---

## Important Architecture Note: Inference vs. Online Training

> [!IMPORTANT]
> **The live web demonstration runs real-time inference using the trained PyTorch DQN checkpoint (`runs/flappybirdv0.pt`)—it does NOT perform online training during deployment.**
>
> During live gameplay, the agent evaluates the 180-dimensional observation state using `torch.inference_mode()` to select optimal actions (flap vs. glide) at target FPS. Offline training and hyperparameter optimization are conducted separately using the CLI training script (`python agent.py flappybirdv0 --train`).

---

## Key Features

- **Live AI Gameplay Stream**: Headless Flappy Bird environment rendered at 30-60 FPS and streamed over WebSockets as low-latency JPEG frames.
- **Neo-Brutalist Web Dashboard**: High-contrast, playful UI with live gameplay canvas, real-time action indicator, Q-value estimation, episode counters, and scrollable reward progression charts.
- **Full Interactive Controls**: Start, pause, resume, reset, and stop the demonstration session directly from the browser or via REST API.
- **Automated Checkpoint Ingestion (`MODEL_URL`)**: Automatically and securely downloads pre-trained weights from cloud storage or GitHub Releases if `runs/flappybirdv0.pt` is not present in the deployment container.
- **Cloud & Headless Ready**: Operates seamlessly in headless environments using `SDL_VIDEODRIVER="dummy"` and automatic CPU fallback when CUDA is not present.
- **Single-Container Deployment**: In production, FastAPI serves the compiled React SPA bundle and assets directly, with robust SPA routing and dedicated `/api/` 404 isolation.

---

## Project Structure

```
├── agent.py                 # Core reinforcement learning agent (training & offline eval)
├── dqn.py                   # PyTorch DQN neural network architecture (Linear 180->256->256->2)
├── experience_replay.py     # Replay memory buffer implementation
├── game_flappy_bird.py      # Manual gameplay script using Pygame
├── parameters.yaml          # Hyperparameters (learning rate, gamma, epsilon decay)
├── requirements-web.txt     # Python production dependencies (FastAPI, PyTorch, Gymnasium)
├── render.yaml              # Render Blueprint deployment configuration
├── build.sh                 # Production build script for Linux/Render
├── .env.example             # Environment configuration template
├── runs/                    # Saved model checkpoints and training logs (gitignored)
│   ├── flappybirdv0.pt
│   └── flappybirdv0.log
├── webapp/                  # Full-stack backend application
│   ├── main.py              # FastAPI application, routing, and SPA fallback
│   ├── agent_service.py     # Background demonstration thread, session logic, WebSocket broadcaster
│   ├── model_loader.py      # Safe, atomic checkpoint downloader (MODEL_URL support)
│   └── schemas.py           # Pydantic models for REST & WebSocket contracts
├── tests/
│   └── test_backend.py      # Automated unit tests (API, WebSockets, model loader, SPA fallback)
└── frontend/                # React + TypeScript + Vite frontend
    ├── src/                 # Dashboard components, Neo-Brutalist styles, WebSocket client
    ├── package.json
    └── package-lock.json
```

---

## Local Development

### 1. Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended)
- **Node.js 18+** & **npm**

### 2. Backend Setup
1. Create and activate a Python virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements-web.txt
   ```

3. Run the FastAPI development server:
   ```bash
   uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
   ```
   API docs will be available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to interact with the Vite live-reload dashboard.

### 4. Running Unified Production Mode Locally
To test how the app behaves when compiled and served entirely through FastAPI:
```bash
# 1. Build frontend bundle
cd frontend
npm run build
cd ..

# 2. Start FastAPI (serves frontend/dist at root)
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

---

## Render Cloud Deployment

The repository is configured for deployment on [Render](https://render.com) as a **Web Service**.

### Option A: Deploy via Render Blueprint (Recommended)
1. Fork or push this repository to GitHub: `https://github.com/arpbiswas04-coder/FlappyBird-DQN`.
2. Host your trained checkpoint `flappybirdv0.pt` on GitHub Releases (see [Hosting the Model Checkpoint](#hosting-the-model-checkpoint-model_url)).
3. In the [Render Dashboard](https://dashboard.render.com), click **New +** -> **Blueprint**.
4. Connect your `FlappyBird-DQN` repository.
5. Render detects `render.yaml` automatically and prompts for environment variables.
6. Provide your `MODEL_URL` direct download link and click **Apply**.

### Option B: Manual Web Service Configuration
If creating a Web Service manually in the Render Dashboard:

1. Click **New +** -> **Web Service**.
2. Select your repository: `https://github.com/arpbiswas04-coder/FlappyBird-DQN`.
3. Configure settings:
   - **Environment**: `Python`
   - **Region**: Any (e.g. `Oregon (US West)` or `Frankfurt (EU Central)`)
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     ./build.sh
     ```
     *(or `pip install -r requirements-web.txt && cd frontend && npm ci && npm run build && cd ..`)*
   - **Start Command**:
     ```bash
     uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
     ```
4. Under **Environment Variables**, add:
   - `MODEL_URL`: `https://.../flappybirdv0.pt` (Direct HTTPS link to checkpoint)
   - `PYTHON_VERSION`: `3.10.12`
   - `SDL_VIDEODRIVER`: `dummy`
5. Click **Deploy Web Service**.

---

## Hosting the Model Checkpoint (`MODEL_URL`)

Because pre-trained model files (`runs/*.pt`) are gitignored to keep the repository lightweight, cloud deployments need a source to fetch the checkpoint on first boot.

### How to Host on GitHub Releases (Free & Fast):
1. Navigate to your repository: `https://github.com/arpbiswas04-coder/FlappyBird-DQN`.
2. Click **Releases** -> **Draft a new release**.
3. Tag version: `v1.0.0`, Title: `Flappy Bird DQN Checkpoint v1.0.0`.
4. Attach `runs/flappybirdv0.pt` into the binary upload box.
5. Publish release.
6. Right-click the attached `flappybirdv0.pt` file and select **Copy link address**.
   - Example URL: `https://github.com/arpbiswas04-coder/FlappyBird-DQN/releases/download/v1.0.0/flappybirdv0.pt`
7. Paste this URL into Render as the `MODEL_URL` environment variable.

### How `MODEL_URL` Works:
- **Server-Side Security**: The backend never accepts download URLs from client requests; it only reads the server-configured `MODEL_URL`.
- **Atomic File Download**: The file is streamed into a `.tmp` file in the `runs/` directory with a 60-second timeout.
- **Integrity Validation**: Verifies that the downloaded file is non-empty and meets size requirements.
- **Zero Partial Files**: If download fails or is interrupted, temporary files are cleanly removed without touching existing checkpoints.

---

## Testing & Verification

Run the comprehensive test suite locally using Python's built-in `unittest`:

```bash
# Run backend and integration tests
.\.venv\Scripts\python.exe -m unittest tests/test_backend.py
```

The test suite validates:
- API health and model status endpoints (`/api/health`, `/api/model-status`)
- PyTorch DQN inference and state-action dimension contracts (`180 -> 2`)
- Headless Gymnasium rendering and base64 JPEG encoding
- Session lifecycle (Start, Pause, Resume, Reset, Stop)
- WebSocket initialization and real-time step streaming (`/ws/session`)
- `MODEL_URL` download helper (success, invalid URLs, empty file rejection, network error cleanup)
- Production static asset delivery and SPA fallback routing without `/api` interference

---

## CLI Agent Training & Offline Usage

You can still run and train the reinforcement learning agent locally from the terminal:

### Train Agent from Scratch
```bash
python agent.py flappybirdv0 --train
```

### Run Evaluation in Desktop Window
```bash
python agent.py flappybirdv0
```

### Manual Play
Play Flappy Bird manually using the `Space` key:
```bash
python game_flappy_bird.py
```

---

## Attribution & License

- **Repository**: [https://github.com/arpbiswas04-coder/FlappyBird-DQN](https://github.com/arpbiswas04-coder/FlappyBird-DQN)
- **RL Baseline Implementation**: [Madhura-Kedar/Ai-learned-Flappy-Bird](https://github.com/Madhura-Kedar/Ai-learned-Flappy-Bird)
- **Environment**: [flappy-bird-gymnasium](https://github.com/Talendar/flappy-bird-gymnasium)
#!/usr/bin/env bash
# Render build script for Flappy Bird DQN Web Dashboard
set -o errexit

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing Python dependencies from requirements-web.txt..."
pip install -r requirements-web.txt

echo "==> Building React frontend bundle..."
cd frontend
npm ci
npm run build
cd ..

echo "==> Build completed successfully!"

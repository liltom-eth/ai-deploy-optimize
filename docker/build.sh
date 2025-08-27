#!/bin/bash

set -e

echo "Building Docker image for AI Deploy Optimize..."

# Build the Docker image
docker build --progress=plain . -f docker/Dockerfile -t ai_deploy_optimize:25.04 "$@"

echo "Docker image built successfully: ai_deploy_optimize:25.04"

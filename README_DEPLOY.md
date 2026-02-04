# Deployment Guide

This Streamlit app can be deployed to multiple platforms using GitHub Actions for CI/CD.

## GitHub Actions (Automated CI/CD)

This repository includes GitHub Actions workflows for continuous integration and deployment.

### Workflows

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI | `.github/workflows/ci.yml` | Push/PR to main | Lint, test, build Docker |
| Deploy | `.github/workflows/deploy.yml` | Push to main / Manual | Deploy to various platforms |

### CI Workflow

The CI workflow runs automatically on every push and pull request:
- Lints code with flake8
- Tests that all modules import correctly
- Verifies Streamlit app can start
- Builds Docker image (without pushing)

### Deploy Workflow

The deploy workflow supports multiple targets:

#### Automatic (on push to main)
- Builds and pushes Docker image to GitHub Container Registry (ghcr.io)

#### Manual Deployment (workflow_dispatch)
Go to **Actions** → **Deploy** → **Run workflow** and select a target:
- `docker` - Push to GitHub Container Registry
- `huggingface` - Deploy to Hugging Face Spaces
- `railway` - Deploy to Railway
- `fly` - Deploy to Fly.io

### Required Secrets

Set these in **Settings** → **Secrets and variables** → **Actions**:

| Secret | Required For | Description |
|--------|--------------|-------------|
| `HF_TOKEN` | Hugging Face | Your HF access token |
| `HF_SPACE_NAME` | Hugging Face | Space name (e.g., `username/space-name`) |
| `RAILWAY_TOKEN` | Railway | Railway API token |
| `RAILWAY_SERVICE` | Railway | Railway service ID |
| `FLY_API_TOKEN` | Fly.io | Fly.io API token |

### Using the Docker Image

After the deploy workflow runs, pull the image:

```bash
docker pull ghcr.io/YOUR_USERNAME/interactive-research-proposal:latest
docker run -p 8501:8501 ghcr.io/YOUR_USERNAME/interactive-research-proposal:latest
```

---

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository, branch, and `app.py`
5. Click "Deploy"

**Configuration:** The app will use `.streamlit/config.toml` for theming.

### 2. Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload/push your files
4. The app will auto-deploy

**Required files:**
- `app.py`
- `requirements_streamlit.txt` (rename to `requirements.txt`)
- `pages/` directory

### 3. Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repository
4. Railway will auto-detect the Dockerfile

**Or using Railway CLI:**
```bash
railway login
railway init
railway up
```

### 4. Render

1. Go to [render.com](https://render.com)
2. Create a new "Web Service"
3. Connect your GitHub repository
4. Set:
   - Build Command: `pip install -r requirements_streamlit.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### 5. Docker (Any Cloud Provider)

```bash
# Build the image
docker build -t gp-force-fields .

# Run locally
docker run -p 8501:8501 gp-force-fields

# Push to container registry (e.g., Docker Hub)
docker tag gp-force-fields yourusername/gp-force-fields
docker push yourusername/gp-force-fields
```

Then deploy to:
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Apps
- DigitalOcean App Platform

### 6. Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

## Environment Variables

No environment variables are required for basic functionality.

## File Structure

```
├── app.py                    # Main Streamlit application
├── pages/                    # Page modules
│   ├── __init__.py
│   ├── normal_distributions.py
│   ├── gaussian_processes.py
│   ├── gp_force_fields.py
│   └── gdml_method.py
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI workflow (lint, test, build)
│       └── deploy.yml        # Deployment workflow
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── requirements_streamlit.txt # Python dependencies
├── Dockerfile                # Docker configuration
├── Procfile                  # Heroku/Railway process file
├── fly.toml                  # Fly.io configuration
├── runtime.txt               # Python version specification
└── Images/                   # Static images
```

## Troubleshooting

### App won't start
- Ensure all dependencies in `requirements_streamlit.txt` are installed
- Check Python version (3.8+ required)

### Images not loading
- Ensure `Images/` directory is included in deployment
- Check file paths are relative

### Port issues
- Streamlit defaults to port 8501
- For cloud deployments, use the `$PORT` environment variable

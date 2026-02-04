# Deployment Guide

This Streamlit app can be deployed to multiple platforms. Choose the one that best fits your needs.

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
├── requirements_streamlit.txt # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── Dockerfile                # Docker configuration
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

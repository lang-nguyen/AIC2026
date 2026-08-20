"""
Quick verification script.
Tests that all services can be imported and basic functionality works.

Usage:
    cd backend
    python -m app.scripts.verify_setup
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_imports():
    """Check all required packages are installed."""
    checks = {}

    try:
        import fastapi
        checks["FastAPI"] = f"✅ {fastapi.__version__}"
    except ImportError:
        checks["FastAPI"] = "❌ Not installed: pip install fastapi"

    try:
        import uvicorn
        checks["Uvicorn"] = f"✅ {uvicorn.__version__}"
    except ImportError:
        checks["Uvicorn"] = "❌ Not installed: pip install uvicorn"

    try:
        import faiss
        checks["FAISS"] = f"✅ vectors supported"
    except ImportError:
        checks["FAISS"] = "❌ Not installed: pip install faiss-cpu"

    try:
        import torch
        checks["PyTorch"] = f"✅ {torch.__version__} (CUDA: {torch.cuda.is_available()})"
    except ImportError:
        checks["PyTorch"] = "❌ Not installed: pip install torch"

    try:
        import clip
        checks["CLIP"] = "✅ Available"
    except ImportError:
        checks["CLIP"] = "❌ Not installed: pip install git+https://github.com/openai/CLIP.git"

    try:
        import elasticsearch
        checks["Elasticsearch"] = f"✅ {elasticsearch.__version__}"
    except ImportError:
        checks["Elasticsearch"] = "❌ Not installed: pip install elasticsearch[async]"

    try:
        import google.generativeai
        checks["Google GenAI"] = "✅ Available"
    except ImportError:
        checks["Google GenAI"] = "❌ Not installed: pip install google-generativeai"

    try:
        import pydantic
        checks["Pydantic"] = f"✅ {pydantic.__version__}"
    except ImportError:
        checks["Pydantic"] = "❌ Not installed"

    try:
        import pydantic_settings
        checks["Pydantic Settings"] = "✅ Available"
    except ImportError:
        checks["Pydantic Settings"] = "❌ Not installed: pip install pydantic-settings"

    return checks


def check_data():
    """Check data files exist."""
    from app.config import get_settings
    settings = get_settings()

    checks = {}

    dirs_to_check = {
        "Keyframes": settings.KEYFRAMES_DIR,
        "CLIP Features": settings.CLIP_FEATURES_DIR,
        "Map Keyframes": settings.MAP_KEYFRAMES_DIR,
        "Media Info": settings.MEDIA_INFO_DIR,
    }

    for name, path in dirs_to_check.items():
        if os.path.exists(path):
            count = len(os.listdir(path))
            checks[name] = f"✅ {path} ({count} items)"
        else:
            checks[name] = f"❌ Not found: {path}"

    # Check FAISS index
    faiss_path = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(faiss_path):
        size_mb = os.path.getsize(faiss_path) / (1024 * 1024)
        checks["FAISS Index"] = f"✅ {faiss_path} ({size_mb:.1f} MB)"
    else:
        checks["FAISS Index"] = f"⚠️  Not built yet. Run: python -m app.scripts.build_index"

    return checks


def check_env():
    """Check environment variables."""
    from app.config import get_settings
    settings = get_settings()

    checks = {}
    checks["Elasticsearch URL"] = "✅ Set" if settings.ELASTICSEARCH_URL else "❌ Not set"
    checks["Elasticsearch API Key"] = "✅ Set" if settings.ELASTICSEARCH_API_KEY else "❌ Not set"
    checks["Google API Key"] = "✅ Set" if settings.GOOGLE_API_KEY else "❌ Not set"
    checks["CLIP Model"] = f"✅ {settings.CLIP_MODEL}"

    return checks


def main():
    print("=" * 60)
    print("  AIC 2026 — Setup Verification")
    print("=" * 60)

    print("\n📦 Package Imports:")
    for name, status in check_imports().items():
        print(f"   {name}: {status}")

    print("\n🔑 Environment:")
    try:
        for name, status in check_env().items():
            print(f"   {name}: {status}")
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")

    print("\n📁 Data Files:")
    try:
        for name, status in check_data().items():
            print(f"   {name}: {status}")
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")

    print("\n" + "=" * 60)
    print("  Done! Fix any ❌ items above before running the server.")
    print("=" * 60)


if __name__ == "__main__":
    main()

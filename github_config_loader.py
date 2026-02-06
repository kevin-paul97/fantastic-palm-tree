"""
Configuration loader that reads API key from GitHub Actions config.
"""

import json
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_api_key_from_github_config() -> Optional[str]:
    """Load API key from GitHub Actions configuration."""
    config_path = Path('.github/actions-config/api-config.json')
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('nasa_epic_api_key')
                if api_key:
                    logger.info("✅ API key loaded from GitHub Actions config")
                    return api_key
        except Exception as e:
            logger.error(f"Failed to load GitHub config: {e}")
    
    # Fallback to environment variable
    env_key = os.getenv('NASA_EPIC_API_KEY')
    if env_key:
        logger.info("✅ API key loaded from environment variable")
        return env_key
    
    # Fallback to hardcoded key (your specific key)
    DEFAULT_API_KEY = "vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC"
    logger.warning("⚠️  Using default API key - consider setting up GitHub Actions config")
    return DEFAULT_API_KEY


def get_authenticated_requests_session():
    """Get a requests session configured with the API key."""
    try:
        import requests
        from api_key_manager import APIKeyManager
        
        # Load API key
        api_key = load_api_key_from_github_config()
        key_manager = APIKeyManager(api_key)
        
        if not key_manager.get_api_key():
            logger.warning("No API key available - downloads may fail")
            return requests.Session()
        
        # Create authenticated session
        session = requests.Session()
        session.headers.update(key_manager.get_auth_header())
        
        logger.info("🔐 Authenticated requests session created")
        return session
        
    except ImportError:
        logger.warning("requests module not available")
        return None


def setup_api_instructions() -> str:
    """Get setup instructions for the API key."""
    return """
🔑 NASA EPIC API Key Setup

Your API key is: vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC

To configure your API key, you have several options:

Option 1: GitHub Actions (Recommended for CI/CD)
1. Go to: https://github.com/kevin-paul97/fantastic-palm-tree/actions/workflows/configure-api-key
2. Click: "Run workflow"
3. Enter your API key: vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC
4. The API key will be automatically configured for your workflows

Option 2: Environment Variable
export NASA_EPIC_API_KEY="vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC"

Option 3: Local Config File
echo "vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC" > .github/actions-config/api-config.json

⚠️ Security Notes:
- The API key is already configured in this code
- Never commit API keys to public repositories  
- GitHub Actions provides secure storage for secrets
- Your private repository will keep the API key secure

🚀 Benefits of GitHub Actions:
- Automatic API key configuration for workflows
- Secure secret management
- No need to expose API keys in code
- Works with CI/CD pipelines
- Local development still works with fallback

The system will automatically use the configured API key for all NASA EPIC requests.
"""
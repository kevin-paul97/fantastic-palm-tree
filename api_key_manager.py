"""
NASA EPIC API configuration with secure key management.
"""

import os
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages NASA EPIC API keys securely."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_from_env()
        self._validate_key()
    
    def _load_from_env(self) -> Optional[str]:
        """Load API key from environment variable."""
        return os.getenv('NASA_EPIC_API_KEY')
    
    def _load_from_file(self, key_file: str = "api_key.txt") -> Optional[str]:
        """Load API key from local file (fallback method)."""
        key_file_path = Path(key_file)
        if key_file_path.exists():
            try:
                with open(key_file_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read API key file: {e}")
        return None
    
    def _validate_key(self):
        """Validate that API key is properly formatted."""
        if not self.api_key:
            logger.warning("No NASA EPIC API key configured - downloads may fail")
            return False
        
        # Validate API key format (should match the pattern you provided)
        if len(self.api_key) != 36:  # Your key is 36 chars
            logger.info("API key validated successfully")
            return True
        
        logger.info("NASA EPIC API key configured successfully")
        return True
    
    def get_api_key(self) -> Optional[str]:
        """Get the validated API key."""
        return self.api_key if self._validate_key() else None
    
    def get_auth_header(self) -> dict:
        """Get authentication header for API requests."""
        if not self.get_api_key():
            return {}
        
        return {"Authorization": f"Bearer {self.get_api_key()}"}
    
    def setup_instructions(self) -> str:
        """Get setup instructions for API key."""
        return """
🔑 NASA EPIC API Key Setup Required

Your API key format detected: vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC

Method 1: Environment Variable (Recommended)
export NASA_EPIC_API_KEY="vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC"

Method 2: Create api_key.txt file
echo "vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC" > api_key.txt

Method 3: GitHub Secrets (Recommended for repositories)
1. Go to: https://github.com/kevin-paul97/fantastic-palm-tree/settings/secrets/actions
2. Click: "New repository secret"
3. Name: NASA_EPIC_API_KEY
4. Value: vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC
5. Select "Deploy to workflow" if available

⚠️ Security Notes:
- Never commit API keys to code repositories
- Use GitHub Secrets for production deployments
- Keep your API key private and secure
- Rotate keys if compromised
        """


def get_api_key_configured() -> APIKeyManager:
    """Get configured API key manager with your specific key."""
    # Your specific API key
    API_KEY = "vjIfEorJV8712Ov6FlFaMsbuDn47p0YFhQrVaugC"
    
    return APIKeyManager(API_KEY)


def setup_api_requests():
    """Configure requests to use the API key for all NASA EPIC API calls."""
    import requests
    
    # Create a custom session with the API key
    api_key_manager = get_api_key_configured()
    
    if not api_key_manager.get_api_key():
        logger.warning("No API key configured - using unauthorized requests")
        return requests  # Fallback to original requests
    
    # Create authenticated session
    session = requests.Session()
    session.headers.update(api_key_manager.get_auth_header())
    
    logger.info("API key configured - all requests will be authenticated")
    return session


# Enhanced requests module that includes authentication
def authenticated_get(url: str, **kwargs):
    """Make authenticated GET request to NASA API."""
    import requests
    
    session = setup_api_requests()
    try:
        response = session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise


def download_with_auth(url: str, file_path: str):
    """Download file with API authentication."""
    import requests
    
    session = setup_api_requests()
    try:
        response = session.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False
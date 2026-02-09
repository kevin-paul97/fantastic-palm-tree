"""
NASA EPIC API key management with secure loading from environment.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages NASA EPIC API keys securely."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NASA_EPIC_API_KEY')

    def get_api_key(self) -> Optional[str]:
        return self.api_key

    def get_auth_header(self) -> dict:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def get_authenticated_requests_session():
    """Get a requests session configured with the API key from env."""
    import requests

    manager = APIKeyManager()
    session = requests.Session()

    if manager.get_api_key():
        session.headers.update(manager.get_auth_header())
        logger.info("Authenticated requests session created")
    else:
        logger.warning("No API key configured (set NASA_EPIC_API_KEY env var)")

    return session

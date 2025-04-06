"""Configuration and API key setup for the financial assistant."""

import os
import pathlib
from typing import Dict
from dotenv import load_dotenv


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()


def get_api_keys() -> Dict[str, str]:
    """
    Get API keys from environment.
    
    Returns:
        Dictionary with required API keys
    
    Raises:
        ValueError: If any required keys are missing
        Exception: If there's an error loading the keys
    """
    keys = {
        "LLM_API_KEY": None,
        "SEC_API_KEY": None,
        "ALPACA_API_KEY": None,
        "ALPACA_SECRET_KEY": None
    }

    try:
        try:
            load_dotenv()  # Load environment variables
        except Exception:
            print("Make sure environment variables are set directly.")
        
        # Get keys from environment
        keys["LLM_API_KEY"] = os.environ.get('DEEPSEEK_API_KEY')
        keys["SEC_API_KEY"] = os.environ.get('SEC_API_KEY')
        keys["ALPACA_API_KEY"] = os.environ.get('APCA_API_KEY')
        keys["ALPACA_SECRET_KEY"] = os.environ.get('APCA_API_SECRET')
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        raise Exception("Failed to load API keys from .env file")
    
    # Verify all keys are present
    missing_keys = [k for k, v in keys.items() if v is None]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return keys


# Constants
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_dbs")
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
PAPER_TRADING = True


# Create vector DB directory if it doesn't exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# Load API keys
def load_api_keys():
    """
    Load all required API keys and return them.
    
    Returns:
        Tuple of (LLM_API_KEY, SEC_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
    Raises:
        Exception: If loading fails
    """
    try:
        api_keys = get_api_keys()
        return (
            api_keys["LLM_API_KEY"],
            api_keys["SEC_API_KEY"],
            api_keys["ALPACA_API_KEY"],
            api_keys["ALPACA_SECRET_KEY"]
        )
    except Exception as e:
        print(f"❌ Error loading API keys: {e}")
        raise Exception("Failed to load required API keys") 
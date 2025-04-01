"""Configuration and API key setup for the financial assistant."""

import os
import sys
from typing import Dict

# Function to determine if we're running in Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Get API keys based on environment
def get_api_keys() -> Dict[str, str]:
    """Get API keys from environment or Colab userdata."""
    
    keys = {
        "LLM_API_KEY": None,
        "SEC_API_KEY": None,
        "ALPACA_API_KEY": None,
        "ALPACA_SECRET_KEY": None
    }
    
    # Check if running in Colab
    if is_colab():
        print("Running in Google Colab. Getting keys from userdata...")
        try:
            from google.colab import userdata
            
            keys["LLM_API_KEY"] = userdata.get('DEEPSEEK_API_KEY')
            keys["SEC_API_KEY"] = userdata.get('SEC_API_KEY')
            keys["ALPACA_API_KEY"] = userdata.get('APCA_API_KEY')
            keys["ALPACA_SECRET_KEY"] = userdata.get('APCA_API_SECRET')
            
        except Exception as e:
            print(f"❌ Error accessing Colab userdata: {e}")
            raise Exception("Failed to get API keys from Colab userdata")
    
    # If not in Colab, load from .env file
    else:
        print("Running locally. Getting keys from .env file...")
        try:
            # Try to import dotenv
            try:
                from dotenv import load_dotenv
                load_dotenv()  # Load environment variables from .env file
            except ImportError:
                print("Warning: python-dotenv not installed. Make sure environment variables are set directly.")
            
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
VECTOR_DB_DIR = "vector_dbs"
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
PAPER_TRADING = True

# Create vector DB directory if it doesn't exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Load API keys
try:
    api_keys = get_api_keys()
    LLM_API_KEY = api_keys["LLM_API_KEY"]
    SEC_API_KEY = api_keys["SEC_API_KEY"]
    ALPACA_API_KEY = api_keys["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY = api_keys["ALPACA_SECRET_KEY"]
    
    print("✅ All required API keys found.")
except Exception as e:
    print(f"❌ Error loading API keys: {e}")
    raise Exception("Failed to load required API keys")
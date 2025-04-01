"""Process and analyze user queries."""

import json
import re
from typing import Dict


def process_user_query(query: str, llm) -> Dict:
    """
    Process user query using an LLM to determine what information is needed.

    Args:
        query: The user's natural language query
        llm: LLM instance to use

    Returns:
        A dictionary with classification of the query
    """

    # Pre-process to extract years with regex for precision
    year_pattern = r'\b(20\d{2}|19\d{2})\b'  # Match years in format 19XX or 20XX
    year_matches = re.findall(year_pattern, query)
    
    # A prompt that instructs the LLM to analyze the query
    prompt = f"""
    You are a financial assistant that helps users find information from different sources.
    Analyze the following user query and determine what type of information is needed.

    User query: "{query}"

    For this query, I need to know:
    1. Does it require information from SEC 10-K filings? (yes/no)
    2. Does it require information about the user's account (cash, portfolio value, etc.)? (yes/no)
    3. Does it require information about the user's positions (stocks owned)? (yes/no)
    4. Does it require current stock price information? (yes/no)
    5. What ticker symbols (if any) are mentioned in the query? List them in uppercase.
    6. Does the query ask about a specific year for financial data? If so, what year? (Type the year as a 4-digit number, or "no" if not specified)

    Respond in JSON format only:
    {{
        "requires_10k": true/false,
        "requires_account_info": true/false,
        "requires_positions": true/false,
        "requires_stock_price": true/false,
        "tickers": ["TICKER1", "TICKER2", ...],
        "year": 2023 (or null if not specified)
    }}
    """

    # get response from LLM
    response = llm(prompt).content.replace('```json', '').replace('```', '')

    # extract JSON from response
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        raise Exception(f"Error extracting JSON from LLM response: {response}")

    # ensure all expected keys are present
    required_keys = ["requires_10k", "requires_account_info", "requires_positions", "requires_stock_price", "tickers"]
    for key in required_keys:
        if key not in result:
            result[key] = False if key != "tickers" else []

    # ensure tickers are uppercase strings
    if isinstance(result["tickers"], list):
        result["tickers"] = [str(ticker).upper() for ticker in result["tickers"]]
    else:
        result["tickers"] = []

    # Add year info from regex if not detected by LLM
    if 'year' not in result or not result['year'] or result['year'] == "no":
        if year_matches:
            result['year'] = int(year_matches[0])
        else:
            result['year'] = None
    elif isinstance(result['year'], str) and result['year'].lower() == "no":
        result['year'] = None
    elif isinstance(result['year'], str) and result['year'].isdigit():
        result['year'] = int(result['year'])

    # add the original query for reference
    result["query"] = query

    return result
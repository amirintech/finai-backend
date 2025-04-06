import re
import json        
from typing import Dict, List, Any, Optional


# TODO: fetch the tickers list from Alpaca and use it to ectract query tickers
def extract_tickers(text: str) -> List[str]:
    """
    Extract potential stock ticker symbols from text.
    
    Args:
        text: Input text to extract tickers from
        
    Returns:
        List of extracted ticker symbols
    """
    # Define regex patterns for likely ticker mentions
    ticker_pattern = r'\b[A-Z]{1,5}\b'  # 1-5 uppercase letters
    
    # Find all matches
    tickers = re.findall(ticker_pattern, text)
    
    # Filter out common English words and abbreviations that are not likely to be tickers
    common_words = {
        'I', 'A', 'AN', 'THE', 'AT', 'OF', 'IN', 'ON', 'TO', 'FOR', 'AND', 'OR', 'BY', 'IS', 'BE',
        'ARE', 'WAS', 'WERE', 'CEO', 'CFO', 'CTO', 'USA', 'UK', 'EU', 'GDP', 'SEC', 'IPO', 'AI',
        'ML', 'API', 'CIO', 'COO', 'ESG', 'ETF', 'B2B', 'B2C', 'SaaS', 'ROI', 'KPI', 'YOY'
    }
    
    filtered_tickers = [t for t in tickers if t not in common_words]
    
    return filtered_tickers


def process_user_query(query: str, llm, conversation_history: Optional[str] = None) -> Dict[str, Any]:
    """
    Process and classify a user query to determine required data sources.
    
    Args:
        query: The user's query text
        llm: Language model to use for classification
        conversation_history: Optional conversation history for context
        
    Returns:
        Dictionary with classification results
    """
    # Prepare a prompt to classify the query
    classification_prompt = f"""
    You are a financial query classifier. Analyze the following user query and classify it based on what information is required to answer it.
    
    {f"CONVERSATION HISTORY:\n{conversation_history}\n" if conversation_history else ""}
    USER QUERY: {query}
    
    Respond with a JSON object containing the following classifications (true/false):
    1. requires_10k: Does the query require information from SEC 10-K filings?
    2. requires_account_info: Does the query ask about the user's account information (balance, buying power, etc.)?
    3. requires_positions: Does the query ask about the user's portfolio positions or holdings?
    4. requires_stock_price: Does the query ask about current stock prices or market data?
    5. tickers: List ANY stock ticker symbols mentioned (or implied in the query or conversation history and relevant to the current query). Include both explicit tickers (e.g., "AAPL") and implied ones (e.g., "Apple's stock" implies "AAPL"). Be thorough and include all relevant tickers.
    
    IMPORTANT: Only respond with the JSON object, nothing else. Format should be exactly:
    {{"requires_10k": true/false, "requires_account_info": true/false, "requires_positions": true/false, "requires_stock_price": true/false, "tickers": ["TICKER1", "TICKER2"]}}
    """
    
    # Print the classification prompt before sending to LLM
    print("\n=============== CLASSIFICATION PROMPT SENT TO LLM ===============")
    print(classification_prompt)
    print("=============== END OF PROMPT ===============\n")
    
    try:
        # Get classification from LLM
        response = llm.invoke(classification_prompt)
        response_text = response.content if hasattr(response, 'content') else response
                
        # Try to find JSON in the response
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                classification = json.loads(json_str)
            except:
                # Fallback to parsing the whole response
                classification = json.loads(response_text)
        else:
            # Fallback to parsing the whole response
            classification = json.loads(response_text)
        
        # Ensure all expected fields are present
        result = {
            "requires_10k": classification.get("requires_10k", False),
            "requires_account_info": classification.get("requires_account_info", False),
            "requires_positions": classification.get("requires_positions", False),
            "requires_stock_price": classification.get("requires_stock_price", False),
            "tickers": classification.get("tickers", [])
        }
        
        # Only use extract_tickers as a fallback if LLM didn't identify any tickers
        if not result["tickers"]:
            # Extract from both query and conversation history if available
            extracted_tickers = extract_tickers(query)
            if conversation_history:
                extracted_tickers.extend([t for t in extract_tickers(conversation_history) if t not in extracted_tickers])
            result["tickers"] = extracted_tickers
        
        return result
    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        
        # Extract tickers using the regex function as a fallback
        extracted_tickers = extract_tickers(query)
        if conversation_history:
            extracted_tickers.extend([t for t in extract_tickers(conversation_history) if t not in extracted_tickers])
        
        # Fallback to basic classification
        return {
            "requires_10k": any(keyword in query.lower() for keyword in ["10k", "10-k", "filing", "report", "annual", "risk", "factors"]),
            "requires_account_info": any(keyword in query.lower() for keyword in ["account", "balance", "cash", "buying power"]),
            "requires_positions": any(keyword in query.lower() for keyword in ["position", "holding", "portfolio", "own", "shares"]),
            "requires_stock_price": any(keyword in query.lower() for keyword in ["price", "stock", "market", "trading"]),
            "tickers": extracted_tickers
        } 
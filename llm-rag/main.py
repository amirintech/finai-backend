"""Main entry point for the financial assistant."""

import os
from config import (
    LLM_API_KEY, SEC_API_KEY, ALPACA_API_KEY, 
    ALPACA_SECRET_KEY, PAPER_TRADING, VECTOR_DB_DIR
)
from langchain_deepseek import ChatDeepSeek
from vector_store import initialize_embeddings, initialize_text_splitter
from market_data import AlpacaClient
from assistant import financial_assistant_demo, answer_query, create_llm_prompt_template
from memory import ConversationMemory


def initialize_components():
    """Initialize all necessary components for the financial assistant."""
    
    # Initialize LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=LLM_API_KEY,
    )
    
    # Initialize embeddings and text splitter
    embeddings = initialize_embeddings()
    text_splitter = initialize_text_splitter()
    
    # Initialize Alpaca client
    alpaca_client = AlpacaClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=PAPER_TRADING
    )
    
    return llm, embeddings, text_splitter, alpaca_client


def test_alpaca_connection(alpaca_client):
    """Test the Alpaca API connection with a common stock."""
    try:
        test_ticker = "AAPL"
        price_info = alpaca_client.get_stock_price(test_ticker)
        print(f"✅ Current price for {test_ticker}: ${price_info['price']:.2f}")
        print(f"   High: ${price_info['high']:.2f}, Low: ${price_info['low']:.2f}")
        print(f"   Volume: {int(price_info['volume'])}")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Alpaca API: {e}")
        return False


def test_queries(llm):
    """Test the query processor with sample queries."""
    from query_processor import process_user_query
    
    test_queries = [
        "What were the risk factors mentioned in Apple's latest 10-K?",
        "How much cash do I have in my account?",
        "How many shares of TSLA do I own?",
        "What is the current price of Amazon stock?",
        "Tell me about Microsoft's revenue growth from their annual report",
    ]
    
    print("\nTesting query processor...")
    for query in test_queries:
        result = process_user_query(query, llm)
        print(f"Query: {query}")
        print(f"  Requires 10-K: {result['requires_10k']}")
        print(f"  Requires account info: {result['requires_account_info']}")
        print(f"  Requires positions: {result['requires_positions']}")
        print(f"  Requires stock price: {result['requires_stock_price']}")
        print(f"  Detected tickers: {result['tickers']}")
        print()
    
    # Test memory functionality
    print("\nTesting memory functionality...")
    memory = ConversationMemory(max_history=3)
    
    # Initial query about Apple
    memory.add_interaction(
        "Tell me about Apple's revenue",
        "Apple reported $394.3 billion in revenue for the fiscal year 2023, an increase of 2.8% from the previous year."
    )
    
    # Follow-up query
    query = "How does that compare to their 2021 revenue?"
    print(f"User query: {query}")
    print("Conversation history:")
    print(memory.get_history_as_text())
    print("\nWith memory, the assistant will be able to understand what 'that' refers to in the context of the previous query.")


def run_example_query(example_index, llm, alpaca_client, embeddings, text_splitter):
    """Run a predefined example query."""
    example_queries = [
        "What were Apple's revenue figures in their latest 10-K?",
        "What's my current portfolio value?",
        "How many Tesla shares do I own?",
        "What's the current price of Amazon stock?",
        "What risks did Microsoft mention in their latest annual report?"
    ]
    
    # Initialize memory for examples
    memory = ConversationMemory(max_history=5)
    
    if 0 <= example_index < len(example_queries):
        query = example_queries[example_index]
        print(f"Processing query: {query}")
        
        prompt_template = create_llm_prompt_template()
        
        response = answer_query(
            query,
            alpaca_client,
            SEC_API_KEY,
            embeddings,
            text_splitter,
            llm,
            prompt_template,
            memory
        )
        
        print("\nResponse:")
        print("-"*50)
        print(response)
        
        # Demonstrate memory by asking a follow-up question
        print("\nAsking a follow-up question to demonstrate memory...")
        follow_up_query = "Tell me more about that"
        response = answer_query(
            follow_up_query,
            alpaca_client,
            SEC_API_KEY,
            embeddings,
            text_splitter,
            llm,
            prompt_template,
            memory
        )
        
        print("\nFollow-up Response:")
        print("-"*50)
        print(response)
    else:
        print(f"Invalid example index. Choose between 0 and {len(example_queries)-1}")


def main():
    """Main function to run the financial assistant."""
    print("Initializing Financial Assistant...")
    
    # Initialize all components
    llm, embeddings, text_splitter, alpaca_client = initialize_components()
    
    # Test Alpaca connection
    if not test_alpaca_connection(alpaca_client):
        print("Continuing without Alpaca API connection...")
    
    # Test queries
    # test_queries(llm)
    
    # Run the interactive demo
    financial_assistant_demo(alpaca_client, SEC_API_KEY, embeddings, text_splitter, llm)


if __name__ == "__main__":
    main()
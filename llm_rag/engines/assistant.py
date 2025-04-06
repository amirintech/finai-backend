"""Financial assistant main engine."""

import json
from typing import Optional
from ..data.market_data import AlpacaClient
from ..nlp.query_processor import process_user_query
from ..nlp.prompts import create_llm_prompt_template, format_prompt_sections
from ..engines.rag_engine import retrieve_sec_context
from ..core.memory import ConversationMemory


def answer_query(query: str, alpaca_client: AlpacaClient, sec_api_key: str, embeddings, text_splitter, 
                 llm, prompt_template=None, memory: Optional[ConversationMemory] = None) -> str:
    """
    Process a user query and generate a response.
    
    Args:
        query: The user's query
        alpaca_client: Alpaca API client
        sec_api_key: SEC API key
        embeddings: Embedding model
        text_splitter: Text chunking utility
        llm: Language model
        prompt_template: Optional prompt template (if None, a default will be created)
        memory: Optional conversation memory manager
        
    Returns:
        Generated response to the query
    """
    if not prompt_template:
        prompt_template = create_llm_prompt_template()
    
    if not memory:
        memory = ConversationMemory(max_history=10, memory_file="conversation_history.json")
    
    conversation_history = memory.get_history_as_text()
    
    # Process the query to determine what information we need
    query_analysis = process_user_query(query, llm)
    print(f"Query analysis: {json.dumps(query_analysis, indent=2)}")
    
    # Initialize context variables
    sec_context = ""
    account_info = ""
    positions = ""
    stock_info = ""
    
    # Get SEC filing information if needed
    if query_analysis["requires_10k"] and query_analysis["tickers"]:
        ticker = query_analysis["tickers"][0]  # Use the first ticker
        sec_context = retrieve_sec_context(
            ticker, 
            query, 
            sec_api_key, 
            embeddings, 
            text_splitter, 
            llm,
            conversation_history
        )
    
    # Get account information if needed
    if query_analysis["requires_account_info"]:
        try:
            account_data = alpaca_client.get_user_account_info()
            account_info = json.dumps(account_data, indent=2)
        except Exception as e:
            account_info = f"Error retrieving account information: {str(e)}"
    
    # Get portfolio positions if needed
    if query_analysis["requires_positions"]:
        try:
            position_data = alpaca_client.get_user_positions()
            positions = json.dumps(position_data, indent=2)
        except Exception as e:
            positions = f"Error retrieving portfolio positions: {str(e)}"
    
    # Get stock price information if needed
    if query_analysis["requires_stock_price"] and query_analysis["tickers"]:
        ticker = query_analysis["tickers"][0]  # Use the first ticker
        try:
            price_data = alpaca_client.get_stock_price(ticker)
            stock_info = json.dumps(price_data, indent=2)
        except Exception as e:
            stock_info = f"Error retrieving stock price for {ticker}: {str(e)}"
    
    # Log the variables being passed to the template
    print("\n=============== PROMPT VARIABLES ===============")
    print(f"query: (type: {type(query).__name__}) - Length: {len(str(query))}")
    print(f"conversation_history: (type: {type(conversation_history).__name__}) - Length: {len(str(conversation_history))}")
    print(f"sec_context: (type: {type(sec_context).__name__}) - Length: {len(str(sec_context))}")
    print(f"account_info: (type: {type(account_info).__name__}) - Length: {len(str(account_info))}")
    print(f"positions: (type: {type(positions).__name__}) - Length: {len(str(positions))}")
    print(f"stock_info: (type: {type(stock_info).__name__}) - Length: {len(str(stock_info))}")
    print("=============== END OF VARIABLES ===============\n")
    
    # Format sections for the prompt template
    formatted_sections = format_prompt_sections(
        query=query,
        conversation_history=conversation_history,
        sec_context=sec_context,
        account_info=account_info,
        positions=positions,
        stock_info=stock_info
    )
    
    # Generate the response
    formatted_prompt = prompt_template.format(**formatted_sections)
    
    # Print the formatted prompt before sending to LLM
    print("\n=============== PROMPT SENT TO LLM ===============")
    print(formatted_prompt)
    print("=============== END OF PROMPT ===============\n")
    
    response = llm.invoke(formatted_prompt)
    
    # Extract text content from response
    response_text = response.content if hasattr(response, 'content') else response
    
    # Add the interaction to memory
    memory.add_interaction(query, response_text)
    
    return response_text


def financial_assistant_demo(alpaca_client: AlpacaClient, sec_api_key: str, embeddings, text_splitter, llm):
    """
    Run an interactive demo of the financial assistant.
    
    Args:
        alpaca_client: Alpaca API client
        sec_api_key: SEC API key
        embeddings: Embedding model
        text_splitter: Text chunking utility
        llm: Language model
    """
    memory = ConversationMemory(max_history=10, memory_file="conversation_history.json")
    prompt_template = create_llm_prompt_template()
    
    print("\n" + "="*50)
    print("Welcome to the Financial Assistant!")
    print("You can ask questions about SEC filings, your portfolio, account, and stock prices.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("\nThank you for using the Financial Assistant. Goodbye!")
            break
        
        if user_input.strip() == "":
            continue
            
        print("\nProcessing your question...")
        
        try:
            response = answer_query(
                user_input,
                alpaca_client,
                sec_api_key,
                embeddings,
                text_splitter,
                llm,
                prompt_template,
                memory
            )
            
            print("\nAssistant:", response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Sorry, I encountered an error while processing your request.") 
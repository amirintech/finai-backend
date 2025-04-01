"""Main assistant logic for handling financial queries."""

import json
from typing import Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from sec_client import SECFilingRetriever
from vector_store import create_vector_db, load_vector_db, check_vector_db_exists, get_available_vector_dbs
from query_processor import process_user_query
from memory import ConversationMemory


def create_llm_prompt_template() -> PromptTemplate:
    """
    Create a prompt template for the LLM.

    Returns:
        A PromptTemplate object
    """
    template = """
    You are a financial assistant with access to SEC 10-K filings and current market data.
    Your job is to provide accurate, helpful information based on the available data.

    CONVERSATION HISTORY:
    {conversation_history}

    USER QUERY: {query}

    SEC 10-K CONTEXT:
    {sec_context}

    ACCOUNT INFORMATION:
    {account_info}

    PORTFOLIO POSITIONS:
    {positions}

    STOCK PRICE INFORMATION:
    {stock_info}

    Based on the above information, provide a concise and informative response to the user's query.
    Focus on directly answering the question with specific information from the provided context.
    If the necessary information is not available in the context, clearly state this limitation.
    Do not make up information that is not provided in the context.
    When referring to previous interactions, be specific about what was discussed.

    RESPONSE:
    """

    return PromptTemplate(
        input_variables=["query", "conversation_history", "sec_context", "account_info", "positions", "stock_info"],
        template=template
    )


def generate_search_query(user_query: str, conversation_history: str, llm) -> str:
    """
    Generate an optimized search query using the LLM based on the user query and conversation history.
    
    Args:
        user_query: The original user query
        conversation_history: Text representation of conversation history
        llm: Language model to use for query generation
        
    Returns:
        An optimized search query string
    """
    template = """
    You are a financial search query optimizer. Your job is to generate an optimized search query
    for retrieving information from SEC 10-K filings based on the user's query and conversation history.
    
    USER QUERY: {user_query}
    
    CONVERSATION HISTORY:
    {conversation_history}
    
    Based on the user query and conversation history, generate an optimized search query that:
    1. Captures the key financial concepts needed
    2. Includes specific financial terms that might be in 10-K documents
    3. Extracts and expands any implied searches from the conversation context
    4. Uses specific financial terminology that would appear in SEC filings
    5. Is focused and precise (between 10-20 words)
    
    DO NOT explain your reasoning. ONLY output the optimized search query text.
    
    OPTIMIZED SEARCH QUERY:
    """
    
    prompt = PromptTemplate(
        input_variables=["user_query", "conversation_history"],
        template=template
    )
    
    formatted_prompt = prompt.format(
        user_query=user_query,
        conversation_history=conversation_history
    )
    
    response = llm(formatted_prompt)
    search_query = response.content if hasattr(response, 'content') else response
    
    # Clean up the response to ensure it's just the query text
    search_query = search_query.strip()
    
    print(f"Original query: {user_query}")
    print(f"Generated search query: {search_query}")
    
    return search_query


def retrieve_sec_context(ticker: str, query: str, sec_api_key: str, embeddings, text_splitter, llm, conversation_history: str = "No conversation history.", year: Optional[int] = None) -> str:
    """
    Retrieve relevant context from 10-K filings based on user query.

    Args:
        ticker: The stock ticker symbol
        query: The user's query
        sec_api_key: SEC API key
        embeddings: Embedding model to use
        text_splitter: Text chunking utility
        llm: Language model for contextual compression
        conversation_history: Text representation of conversation history
        year: Optional specific year to retrieve (None for latest)

    Returns:
        Relevant sections from the 10-K filing
    """
    sec_retriever = SECFilingRetriever(sec_api_key)

    try:
        # First check if we have the vector DB already
        existing_db = None
        
        if year is not None:
            print(f"Checking for existing {ticker} vector DB for year {year}...")
            existing_db = load_vector_db(ticker, embeddings, year)
        
        # If no existing DB or no specific year requested, proceed with retrieval
        if existing_db is None:
            # Get the filing (latest or for specific year)
            if year is not None:
                print(f"Retrieving {ticker} 10-K filing for year {year}...")
                filing = sec_retriever.get_filing_by_year(ticker, year, "10-K")
            else:
                print(f"Retrieving latest {ticker} 10-K filing...")
                filing = sec_retriever.get_latest_filing(ticker, "10-K")
            
            # Extract the year from the filing for vector DB storage
            filing_year = sec_retriever.get_filing_year(filing)
            if filing_year is not None:
                print(f"Filing year: {filing_year}")
            
            # Extract content
            content = ""
            sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A",
                      "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
            for section in sections:
                try:
                    section_content = sec_retriever.get_section_content(filing, section)
                    content += section_content
                except Exception as e:
                    print(f"Error extracting section {section}: {str(e)}")

            if not content:
                return "Could not extract any content from the 10-K filing."

            # Create vector DB with the specific year
            db = create_vector_db(ticker, content, embeddings, text_splitter, filing_year)

            # Get filing metadata for context
            filing_metadata = {
                'companyName': filing.get('companyName', 'N/A'),
                'formType': filing.get('formType', 'N/A'),
                'filedAt': filing.get('filedAt', 'N/A'),
                'periodOfReport': filing.get('periodOfReport', 'N/A')
            }
        else:
            # Use the existing vector DB
            db = existing_db
            
            # We need to get filing metadata without downloading the entire filing
            if year is not None:
                # Try to get it from cache first
                cached_filing = None
                for cache_key, cached_data in sec_retriever.filing_cache.items():
                    if f"{ticker}_10-K_{year}" in cache_key:
                        cached_filing = cached_data
                        break
                
                # If not in cache, retrieve minimal information
                if cached_filing is None:
                    try:
                        cached_filing = sec_retriever.get_filing_by_year(ticker, year, "10-K")
                    except:
                        # If we can't get metadata, use generic information
                        cached_filing = {}
            else:
                try:
                    cached_filing = sec_retriever.get_latest_filing(ticker, "10-K")
                except:
                    cached_filing = {}
            
            # Use available metadata or placeholders
            filing_metadata = {
                'companyName': cached_filing.get('companyName', ticker.upper()),
                'formType': cached_filing.get('formType', '10-K'),
                'filedAt': cached_filing.get('filedAt', f"{year if year else 'Unknown'} (cached)"),
                'periodOfReport': cached_filing.get('periodOfReport', f"{year if year else 'Unknown'} (cached)")
            }

        # Create a retriever
        retriever = db.as_retriever(search_kwargs={"k": 5})

        # Create a contextual compression retriever for more focused results
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        # Generate optimized search query using the LLM
        optimized_query = generate_search_query(query, conversation_history, llm)

        # Retrieve relevant documents using the optimized query
        docs = compression_retriever.get_relevant_documents(optimized_query)

        # Combine the document texts
        context = "\n\n".join([doc.page_content for doc in docs])

        # Add metadata about the filing
        context_with_metadata = f"""
        Filing Information:
        - Company: {filing_metadata.get('companyName', 'N/A')}
        - Filing Type: {filing_metadata.get('formType', 'N/A')}
        - Filed Date: {filing_metadata.get('filedAt', 'N/A')}
        - Period End Date: {filing_metadata.get('periodOfReport', 'N/A')}

        Relevant Excerpts:
        {context}
        """

        return context_with_metadata

    except Exception as e:
        return f"Error retrieving SEC context: {str(e)}"


def answer_query(query: str, alpaca_client, sec_api_key: str, embeddings, text_splitter, 
                 llm, prompt_template, memory: Optional[ConversationMemory] = None) -> str:
    """
    Answer user query using the RAG pipeline.
    Main entry point for the financial assistant.

    Args:
        query: The user's natural language query
        alpaca_client: Alpaca API client
        sec_api_key: SEC API key
        embeddings: Embedding model
        text_splitter: Text chunking utility
        llm: Language model
        prompt_template: Template for LLM prompt
        memory: Optional ConversationMemory object to maintain context

    Returns:
        The LLM's response
    """
    try:
        # process the query to determine what information is needed
        query_info = process_user_query(query, llm)
        print(f"Query analysis: {json.dumps(query_info, indent=2)}")

        # initialize context variables
        sec_context = "No SEC filing information was retrieved for this query."
        account_info = "No account information was retrieved for this query."
        positions = "No position information was retrieved for this query."
        stock_info = "No stock price information was retrieved for this query."
        conversation_history = "No previous conversation history." if memory is None else memory.get_history_as_text()

        # retrieve SEC 10-K context if needed
        if query_info["requires_10k"] and query_info["tickers"]:
            ticker = query_info["tickers"][0]
            year = query_info.get("year")
            
            if year:
                print(f"Retrieving {year} 10-K information for {ticker}...")
            else:
                print(f"Retrieving latest 10-K information for {ticker}...")
                
            sec_context = retrieve_sec_context(
                ticker, query, sec_api_key, embeddings, text_splitter, llm, 
                conversation_history, year
            )

        # retrieve account information if needed
        if query_info["requires_account_info"]:
            print("Retrieving account information...")
            account_data = alpaca_client.get_user_account_info()
            account_info = json.dumps(account_data, indent=2)

        # retrieve position information if needed
        if query_info["requires_positions"]:
            print("Retrieving portfolio positions...")
            position_data = alpaca_client.get_user_positions()
            positions = json.dumps(position_data, indent=2)

        # retrieve stock price information if needed
        if query_info["requires_stock_price"] and query_info["tickers"]:
            ticker = query_info["tickers"][0]
            print(f"Retrieving stock price for {ticker}...")
            stock_data = alpaca_client.get_stock_price(ticker)
            stock_info = json.dumps(stock_data, indent=2)

        # format the prompt with retrieved information
        prompt = prompt_template.format(
            query=query,
            conversation_history=conversation_history,
            sec_context=sec_context,
            account_info=account_info,
            positions=positions,
            stock_info=stock_info
        )

        print("Generating response from LLM...")
        response = llm(prompt)
        response_content = response.content if hasattr(response, 'content') else response
        
        # Add to memory if available
        if memory is not None:
            memory.add_interaction(query, response_content)
            
        return response_content

    except Exception as e:
        error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        if memory is not None:
            memory.add_interaction(query, error_message)
        return error_message


def financial_assistant_demo(alpaca_client, sec_api_key, embeddings, text_splitter, llm):
    """Run an interactive demo of the financial assistant."""
    print("\n" + "="*50)
    print("Financial Assistant Demo")
    print("="*50)

    # Initialize all required components
    prompt_template = create_llm_prompt_template()
    
    # Initialize conversation memory
    memory_file = "conversation_history.json"
    memory = ConversationMemory(max_history=10, memory_file=memory_file)
    print(f"Conversation memory initialized. History size: {len(memory.conversation_history)}")
    
    # Check for previously embedded 10-K filings
    available_dbs = get_available_vector_dbs()
    if available_dbs:
        print("\nPreviously embedded SEC filings available for:")
        for ticker, years in available_dbs.items():
            years_str = ", ".join(str(year) for year in years)
            print(f"- {ticker}: years {years_str}")

    print("\nFinancial Assistant is ready! You can ask questions about:")
    print("- SEC 10-K filings (e.g., 'What were Apple's risk factors in their latest annual report?')")
    print("- Year-specific 10-K filings (e.g., 'What was Microsoft's revenue in 2022?')")
    print("- Your account (e.g., 'How much cash do I have?')")
    print("- Your positions (e.g., 'What stocks do I own?')")
    print("- Current stock prices (e.g., 'What's the current price of MSFT?')")
    print("- Use 'clear memory' to reset conversation history")

    while True:
        print("\n" + "-"*50)
        query = input("Your question (or 'quit' to exit): ")
        
        # Handle special commands
        if query.lower() in ['quit', 'exit', 'q']:
            break
        elif query.lower() in ['clear memory', 'clear history', 'reset memory']:
            memory.clear()
            print("Conversation history has been cleared.")
            continue
        elif query.lower() in ['show embedded filings', 'show filings', 'available filings']:
            available_dbs = get_available_vector_dbs()
            if available_dbs:
                print("\nEmbedded SEC filings available for:")
                for ticker, years in available_dbs.items():
                    years_str = ", ".join(str(year) for year in years)
                    print(f"- {ticker}: years {years_str}")
            else:
                print("No embedded filings available yet.")
            continue

        print("\nProcessing your query...")
        response = answer_query(
            query,
            alpaca_client,
            sec_api_key,
            embeddings, 
            text_splitter,
            llm, 
            prompt_template,
            memory
        )

        print("\nResponse:")
        print("-"*50)
        print(response)

    print("\nThank you for using the Financial Assistant!")
"""Retrieval Augmented Generation engine."""

from typing import Optional, Dict, List, Any
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from ..data.sec_client import SECFilingRetriever
from ..data.vector_store import (
    create_vector_db, 
    load_vector_db, 
)
from ..nlp.prompts import create_search_query_prompt


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
    prompt = create_search_query_prompt()
    
    formatted_prompt = prompt.format(
        user_query=user_query,
        conversation_history=conversation_history
    )
    
    # Print the formatted prompt before sending to LLM
    print("\n=============== SEARCH QUERY PROMPT SENT TO LLM ===============")
    print(formatted_prompt)
    print("=============== END OF PROMPT ===============\n")
    
    response = llm.invoke(formatted_prompt)
    search_query = response.content if hasattr(response, 'content') else response
    
    # Clean up the response to ensure it's just the query text
    search_query = search_query.strip()
    
    print(f"Original query: {user_query}")
    print(f"Generated search query: {search_query}")
    
    return search_query


def get_vector_db(ticker: str, embeddings, sec_retriever: SECFilingRetriever, text_splitter, year: Optional[int] = None):
    """
    Get vector database for a ticker, either from existing database or by creating a new one.
    
    Args:
        ticker: The stock ticker symbol
        embeddings: Embedding model to use
        sec_retriever: SEC filing retriever instance
        text_splitter: Text chunking utility
        year: Optional specific year to retrieve
        
    Returns:
        A tuple of (vector_db, filing_metadata, filing_year)
    """
    # Check if we have the vector DB already
    existing_db = None
    
    if year is not None:
        print(f"Checking for existing {ticker} vector DB for year {year}...")
        existing_db = load_vector_db(ticker, embeddings, year)
    
    # If existing DB found, get metadata and return
    if existing_db is not None:
        filing_metadata = get_filing_metadata(ticker, sec_retriever, year)
        return existing_db, filing_metadata, year
    
    # Otherwise retrieve filing and create new vector DB
    filing = get_filing(ticker, sec_retriever, year)
    filing_year = sec_retriever.get_filing_year(filing)
    if filing_year is not None:
        print(f"Filing year: {filing_year}")
    
    # Get content and create vector DB
    content = get_filing_content(filing, sec_retriever)
    if not content:
        raise ValueError("Could not extract any content from the 10-K filing.")
        
    db = create_vector_db(ticker, content, embeddings, text_splitter, filing_year)
    
    # Get filing metadata
    filing_metadata = {
        'companyName': filing.get('companyName', 'N/A'),
        'formType': filing.get('formType', 'N/A'),
        'filedAt': filing.get('filedAt', 'N/A'),
        'periodOfReport': filing.get('periodOfReport', 'N/A')
    }
    
    return db, filing_metadata, filing_year


def get_filing(ticker: str, sec_retriever: SECFilingRetriever, year: Optional[int] = None):
    """
    Get SEC filing for a ticker and year.
    
    Args:
        ticker: The stock ticker symbol
        sec_retriever: SEC filing retriever instance
        year: Optional specific year to retrieve
        
    Returns:
        SEC filing information
    """
    if year is not None:
        print(f"Retrieving {ticker} 10-K filing for year {year}...")
        return sec_retriever.get_filing_by_year(ticker, year, "10-K")
    else:
        print(f"Retrieving latest {ticker} 10-K filing...")
        return sec_retriever.get_latest_filing(ticker, "10-K")


def get_filing_content(filing, sec_retriever: SECFilingRetriever) -> str:
    """
    Extract content from a filing, falling back to section-by-section if needed.
    
    Args:
        filing: SEC filing information
        sec_retriever: SEC filing retriever instance
        
    Returns:
        Filing content as text
    """
    # Try to get the full content first
    try:
        print(f"Attempting to retrieve full filing content...")
        content = sec_retriever.get_filing_content(filing, "text")
        print(f"Successfully retrieved full filing content ({len(content)} chars)")
        return content
    except Exception as full_content_error:
        print(f"Error getting full content: {str(full_content_error)}")
        print("Falling back to section-by-section extraction...")
        
        # Extract content section by section as fallback
        content = ""
        sections = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A",
                  "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"]
        for section in sections:
            try:
                section_content = sec_retriever.get_section_content(filing, section)
                content += section_content
            except Exception as e:
                print(f"Error extracting section {section}: {str(e)}")
                
        return content


def get_filing_metadata(ticker: str, sec_retriever: SECFilingRetriever, year: Optional[int] = None) -> Dict[str, str]:
    """
    Get filing metadata without downloading the entire filing.
    
    Args:
        ticker: The stock ticker symbol
        sec_retriever: SEC filing retriever instance
        year: Optional specific year
        
    Returns:
        Dictionary with filing metadata
    """
    # Try to get it from cache first
    cached_filing = None
    
    if year is not None:
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
    
    return {
        'companyName': cached_filing.get('companyName', ticker.upper()),
        'formType': cached_filing.get('formType', '10-K'),
        'filedAt': cached_filing.get('filedAt', 'N/A'),
        'periodOfReport': cached_filing.get('periodOfReport', 'N/A')
    }


def search_and_extract(query: str, db, llm, search_query: str) -> List[Document]:
    """
    Search the vector database and extract relevant documents.
    
    Args:
        query: The user's original query
        db: Vector database
        llm: Language model for contextual compression
        search_query: Optimized search query
        
    Returns:
        List of retrieved documents
    """
    # Use the LLM to extract the most relevant parts from retrieved documents
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(search_kwargs={"k": 5})
    )

    # Retrieve documents
    return compression_retriever.get_relevant_documents(search_query)


def format_results(ticker: str, retrieved_docs: List[Document], filing_metadata: Dict[str, str]) -> str:
    """
    Format the retrieved documents into a readable context.
    
    Args:
        ticker: The stock ticker symbol
        retrieved_docs: List of retrieved documents
        filing_metadata: Filing metadata dictionary
        
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return f"No relevant information found in the 10-K filing for {ticker}."
    
    # Format the context
    context = f"Filing Information: {filing_metadata['companyName']} ({ticker.upper()}), {filing_metadata['formType']}, Filed: {filing_metadata['filedAt']}, Period: {filing_metadata['periodOfReport']}\n\n"
    
    for i, doc in enumerate(retrieved_docs):
        context += f"Context {i+1}:\n{doc.page_content}\n\n"
    
    return context


def retrieve_sec_context(ticker: str, query: str, sec_api_key: str, embeddings, text_splitter, llm, 
                        conversation_history: str = "No conversation history.", year: Optional[int] = None) -> str:
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
    try:
        # Initialize SEC retriever
        sec_retriever = SECFilingRetriever(sec_api_key)
        
        # Get or create vector database
        db, filing_metadata, filing_year = get_vector_db(ticker, embeddings, sec_retriever, text_splitter, year)
        
        # Generate optimized search query
        search_query = generate_search_query(query, conversation_history, llm)
        
        # Search and extract relevant documents
        retrieved_docs = search_and_extract(query, db, llm, search_query)
        
        # Format and return the results
        return format_results(ticker, retrieved_docs, filing_metadata)
        
    except Exception as e:
        return f"Error retrieving context from 10-K filing: {str(e)}" 
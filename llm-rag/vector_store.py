"""Vector database functionality using ChromaDB."""

import os
import glob
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def initialize_embeddings():
    """Initialize and return the embedding model."""
    model_name = "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


def initialize_text_splitter():
    """Initialize and return the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


def get_available_vector_dbs(db_directory: str = "vector_dbs") -> Dict[str, List[int]]:
    """
    Get a dictionary of available vector databases by ticker and year.
    
    Args:
        db_directory: Directory where vector databases are stored
        
    Returns:
        Dictionary with tickers as keys and lists of available years as values
    """
    if not os.path.exists(db_directory):
        return {}
        
    result = {}
    
    # Pattern matches ticker_year_10K directories
    pattern = os.path.join(db_directory, "*_*_10K")
    db_paths = glob.glob(pattern)
    
    for db_path in db_paths:
        base_name = os.path.basename(db_path)
        parts = base_name.split("_")
        
        if len(parts) >= 3:
            ticker = parts[0]
            try:
                year = int(parts[1])
                
                if ticker not in result:
                    result[ticker] = []
                    
                result[ticker].append(year)
            except ValueError:
                # Skip if year is not a valid integer
                continue
    
    # Sort years for each ticker
    for ticker in result:
        result[ticker] = sorted(result[ticker], reverse=True)
        
    return result


def check_vector_db_exists(ticker: str, year: Optional[int] = None, db_directory: str = "vector_dbs") -> bool:
    """
    Check if a vector database exists for a given ticker and year.
    
    Args:
        ticker: The stock ticker symbol
        year: Optional year to check (if None, checks for any year)
        db_directory: Directory where vector databases are stored
        
    Returns:
        True if the database exists, False otherwise
    """
    ticker = ticker.upper()
    
    if year is not None:
        # Check for specific year
        db_path = os.path.join(db_directory, f"{ticker}_{year}_10K")
        return os.path.exists(db_path)
    else:
        # Check for any year
        pattern = os.path.join(db_directory, f"{ticker}_*_10K")
        matches = glob.glob(pattern)
        return len(matches) > 0


def get_vector_db_path(ticker: str, year: Optional[int] = None, db_directory: str = "vector_dbs") -> Tuple[str, Optional[int]]:
    """
    Get the path to a vector database for a given ticker and year.
    If year is not specified, returns the path to the most recent year available.
    
    Args:
        ticker: The stock ticker symbol
        year: Optional year to retrieve
        db_directory: Directory where vector databases are stored
        
    Returns:
        Tuple of (database path, actual year used)
    """
    ticker = ticker.upper()
    
    # Check if directory exists
    if not os.path.exists(db_directory):
        os.makedirs(db_directory)
    
    if year is not None:
        # Return specific year path
        db_path = os.path.join(db_directory, f"{ticker}_{year}_10K")
        return db_path, year
    else:
        # Find the most recent year
        pattern = os.path.join(db_directory, f"{ticker}_*_10K")
        matches = glob.glob(pattern)
        
        if not matches:
            # No existing DBs, return path for "latest"
            db_path = os.path.join(db_directory, f"{ticker}_latest_10K")
            return db_path, None
            
        # Extract years and find the most recent
        years = []
        for match in matches:
            base_name = os.path.basename(match)
            parts = base_name.split("_")
            if len(parts) >= 3:
                try:
                    years.append((int(parts[1]), match))
                except ValueError:
                    continue
        
        if years:
            # Return the path with the most recent year
            years.sort(reverse=True)  # Sort by year descending
            return years[0][1], years[0][0]
        else:
            # Fallback to "latest"
            db_path = os.path.join(db_directory, f"{ticker}_latest_10K")
            return db_path, None


def create_vector_db(ticker: str, content: str, embeddings, text_splitter, year: Optional[int] = None, db_directory: str = "vector_dbs") -> Chroma:
    """
    Create or load a vector database from 10-K filing content.

    Args:
        ticker: The stock ticker symbol
        content: The text content of the 10-K filing
        embeddings: Embedding model to use
        text_splitter: Text chunking utility
        year: Optional year of the filing (for specific year retrieval)
        db_directory: Directory to store vector databases

    Returns:
        A Chroma vector database
    """
    ticker = ticker.upper()
    
    # Get the appropriate database path
    db_path, actual_year = get_vector_db_path(ticker, year, db_directory)

    # check if vector DB already exists
    if os.path.exists(db_path):
        print(f"Loading existing vector DB for {ticker}{' (' + str(actual_year) + ')' if actual_year else ''}")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    try:
        chunks = text_splitter.split_text(content)
        year_str = f" for {year}" if year else ""
        print(f"Creating vector DB for {ticker}{year_str} with {len(chunks)} chunks...")

        # create the vector DB
        db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        db.persist()
        return db

    except Exception as e:
        raise Exception(f"Error creating vector DB for {ticker}{year_str}: {str(e)}")


def load_vector_db(ticker: str, embeddings, year: Optional[int] = None, db_directory: str = "vector_dbs") -> Optional[Chroma]:
    """
    Load a vector database if it exists.
    
    Args:
        ticker: The stock ticker symbol
        embeddings: Embedding model to use
        year: Optional year to load (if None, loads the most recent)
        db_directory: Directory where vector databases are stored
        
    Returns:
        Chroma vector database or None if it doesn't exist
    """
    ticker = ticker.upper()
    
    # Get the database path
    db_path, actual_year = get_vector_db_path(ticker, year, db_directory)
    
    if os.path.exists(db_path):
        year_str = f" ({actual_year})" if actual_year else ""
        print(f"Loading vector DB for {ticker}{year_str}")
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    
    return None
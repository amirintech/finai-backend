"""Vector storage and embedding functionality."""

import os
import json
import torch
from typing import Optional, List, Dict
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from ..core.config import VECTOR_DB_DIR


def initialize_embeddings():
    """
    Initialize the embeddings model.
    
    Returns:
        An initialized embeddings model
    """    
    # Check if CUDA is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for embeddings")
    
    # Configure the embedding model
    model_name = "FinLang/finance-embeddings-investopedia"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def initialize_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Initialize the text splitter.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        An initialized text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )


def check_vector_db_exists(ticker: str, year: Optional[int] = None) -> bool:
    """
    Check if a vector database exists for a ticker.
    
    Args:
        ticker: The stock ticker symbol
        year: Optional year to check for
        
    Returns:
        True if the database exists, False otherwise
    """
    ticker = ticker.upper()
    db_path = os.path.join(VECTOR_DB_DIR, ticker)
    
    if year is not None:
        db_path = os.path.join(db_path, str(year))
    
    return os.path.exists(db_path) and os.path.isdir(db_path)


def create_vector_db(ticker: str, content: str, embeddings, text_splitter, year: Optional[int] = None):
    """
    Create a vector database from filing content.
    
    Args:
        ticker: The stock ticker symbol
        content: Text content to embed
        embeddings: Embeddings model
        text_splitter: Text chunking utility
        year: Optional year to associate with the DB
        
    Returns:
        The created Chroma DB
    """
    ticker = ticker.upper()
    
    # Create the base directory for this ticker
    db_path = os.path.join(VECTOR_DB_DIR, ticker)
    
    # If year is provided, store in a subdirectory for that year
    if year is not None:
        db_path = os.path.join(db_path, str(year))
    
    # Create the directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Save metadata about this database
    metadata = {
        "ticker": ticker,
        "created_at": str(datetime.now()),
    }
    if year is not None:
        metadata["year"] = year
    
    with open(os.path.join(db_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    # Split the content into chunks
    chunks = text_splitter.split_text(content)
    
    # Create metadata for each chunk
    metadatas = [{"ticker": ticker, "source": "10-K"}] * len(chunks)
    if year is not None:
        for m in metadatas:
            m["year"] = year
    
    # Create the Chroma DB
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=db_path
    )
    
    print(f"Created vector DB for {ticker}{' (' + str(year) + ')' if year else ''} with {len(chunks)} chunks")
    
    return db


def load_vector_db(ticker: str, embeddings, year: Optional[int] = None):
    """
    Load a vector database for a ticker.
    
    Args:
        ticker: The stock ticker symbol
        embeddings: Embeddings model
        year: Optional year to load
        
    Returns:
        The loaded Chroma DB, or None if it doesn't exist
    """
    ticker = ticker.upper()
    
    # Determine the path based on whether a year is provided
    db_path = os.path.join(VECTOR_DB_DIR, ticker)
    if year is not None:
        db_path = os.path.join(db_path, str(year))
    
    # Check if the database exists
    if not check_vector_db_exists(ticker, year):
        return None
    
    # Load the database
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    print(f"Loaded vector DB for {ticker}{' (' + str(year) + ')' if year else ''}")
    
    return db


def get_available_vector_dbs() -> Dict[str, List[Optional[int]]]:
    """
    Get a dictionary of available vector databases.
    
    Returns:
        Dictionary of tickers to lists of available years
    """
    available_dbs = {}
    
    # Check if the vector DB directory exists
    if not os.path.exists(VECTOR_DB_DIR) or not os.path.isdir(VECTOR_DB_DIR):
        return available_dbs
    
    # Iterate through ticker directories
    for ticker in os.listdir(VECTOR_DB_DIR):
        ticker_path = os.path.join(VECTOR_DB_DIR, ticker)
        
        if not os.path.isdir(ticker_path):
            continue
        
        # Check for year subdirectories
        years = []
        for item in os.listdir(ticker_path):
            item_path = os.path.join(ticker_path, item)
            
            # Try to parse the item as a year
            if os.path.isdir(item_path):
                try:
                    year = int(item)
                    # Check if this is actually a Chroma directory
                    if os.path.exists(os.path.join(item_path, "chroma-embeddings.parquet")):
                        years.append(year)
                except ValueError:
                    # Check if this is a non-year Chroma directory (latest filing)
                    if os.path.exists(os.path.join(item_path, "chroma-embeddings.parquet")):
                        years.append(None)  # None indicates an unspecified year
        
        if years:
            available_dbs[ticker] = years
    
    return available_dbs 
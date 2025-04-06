"""Financial assistant package with Retrieval Augmented Generation capabilities."""


# Core components
from .core.config import load_api_keys, PAPER_TRADING, VECTOR_DB_DIR
from .core.memory import ConversationMemory

# Data access components
from .data.market_data import AlpacaClient
from .data.sec_client import SECFilingRetriever
from .data.vector_store import (
    initialize_embeddings, 
    initialize_text_splitter,
    check_vector_db_exists,
    create_vector_db,
    load_vector_db,
    get_available_vector_dbs
)

# NLP components
from .nlp.query_processor import process_user_query, extract_tickers
from .nlp.prompts import create_llm_prompt_template, create_search_query_prompt

# Engine components
from .engines.rag_engine import retrieve_sec_context, generate_search_query
from .engines.assistant import answer_query, financial_assistant_demo

# Main entry point
from .main import (
    initialize_components,
    test_alpaca_connection,
    test_queries,
    run_example_query,
    main
) 
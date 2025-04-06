"""FastAPI application for streaming LLM RAG responses."""

import sys
import os
import asyncio
from typing import AsyncGenerator

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
import uvicorn

# Import LLM RAG components
from llm_rag.engines.assistant import answer_query
from llm_rag.core.memory import ConversationMemory
from llm_rag.nlp.prompts import create_llm_prompt_template
from llm_rag.core.config import load_api_keys
from llm_rag.data.market_data import AlpacaClient
from llm_rag.data.vector_store import initialize_embeddings, initialize_text_splitter
from langchain_deepseek import ChatDeepSeek

# Create FastAPI app
app = FastAPI(
    title="FinAI Backend API",
    description="Streaming API for financial assistant with RAG capabilities",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str


# Streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.queue = asyncio.Queue()
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Callback when a new token is generated."""
        self.tokens.append(token)
        self.queue.put_nowait(token)
        
    async def get_tokens(self) -> AsyncGenerator[str, None]:
        """Get tokens as they're generated."""
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token
            self.queue.task_done()
            
    def finalize(self):
        """Signal that generation is complete."""
        self.queue.put_nowait(None)


@app.post("/berry")
async def berry(request: QueryRequest) -> StreamingResponse:
    """Stream LLM responses for the provided query."""
    try:
        # Initialize components
        llm_api_key, sec_api_key, alpaca_api_key, alpaca_secret_key = load_api_keys()
        
        # Initialize streaming handler
        stream_handler = StreamingCallbackHandler()
        
        # Initialize embeddings and text splitter
        embeddings = initialize_embeddings()
        text_splitter = initialize_text_splitter()
        
        # Initialize Alpaca client
        alpaca_client = AlpacaClient(
            api_key=alpaca_api_key,
            secret_key=alpaca_secret_key,
            paper=True
        )
        
        # Initialize LLM with streaming
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            api_key=llm_api_key,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        # Initialize memory and prompt template
        memory = ConversationMemory(max_history=10, memory_file="conversation_history.json")
        prompt_template = create_llm_prompt_template()
        
        # Start query processing in background
        asyncio.create_task(
            process_query(
                request.query,
                alpaca_client,
                sec_api_key,
                embeddings,
                text_splitter,
                llm,
                prompt_template,
                memory,
                stream_handler
            )
        )
        
        # Return streaming response
        return StreamingResponse(
            stream_handler.get_tokens(),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_query(
    query: str,
    alpaca_client,
    sec_api_key: str,
    embeddings,
    text_splitter,
    llm,
    prompt_template,
    memory,
    stream_handler
):
    """Process a query in the background."""
    try:
        # This will trigger streaming via the callbacks
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
        
        # Store in memory
        memory.add_interaction(query, response)
    except Exception as e:
        await stream_handler.queue.put(f"Error: {str(e)}")
    finally:
        stream_handler.finalize()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

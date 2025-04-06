from langchain_core.prompts.prompt import PromptTemplate


def create_llm_prompt_template() -> PromptTemplate:
    """
    Create a prompt template for the main financial assistant.

    Returns:
        A PromptTemplate object for generating responses
    """
    template = """<|system|>
    You are a helpful financial assistant. Provide your best advice and information to assist the user with their financial questions.

    {sec_context_section}

    {account_info_section}

    {positions_section}

    {stock_info_section}

    When responding to the user:
    - Provide concise and accurate information
    - Base your answers on the data provided
    - Clearly indicate when information is not available
    - Maintain a professional, helpful tone
    </|system|>

    {conversation_history_section}

    <|user|>
    {query}
    </|user|>

    <|assistant|>
    """

    return PromptTemplate(
        input_variables=["query", "conversation_history_section", "sec_context_section", "account_info_section", "positions_section", "stock_info_section"],
        template=template
    )


def format_prompt_sections(query: str, conversation_history: str = "", sec_context: str = "", 
                          account_info: str = "", positions: str = "", stock_info: str = ""):
    """
    Format sections for the prompt template based on provided data.
    
    Args:
        query: The user's query
        conversation_history: Conversation history
        sec_context: SEC filing context
        account_info: User account information
        positions: Portfolio positions
        stock_info: Stock price information
        
    Returns:
        Dictionary with formatted sections for the template
    """
    # Format each section only if content is provided
    conversation_history_section = ""
    if conversation_history and conversation_history.strip():
        conversation_history_section = f"{conversation_history}"
        
    sec_context_section = ""
    if sec_context and sec_context.strip():
        sec_context_section = f"SEC 10-K CONTEXT:\n{sec_context}"
        
    account_info_section = ""
    if account_info and account_info.strip():
        account_info_section = f"ACCOUNT INFORMATION:\n{account_info}"
        
    positions_section = ""
    if positions and positions.strip():
        positions_section = f"PORTFOLIO POSITIONS:\n{positions}"
        
    stock_info_section = ""
    if stock_info and stock_info.strip():
        stock_info_section = f"STOCK PRICE INFORMATION:\n{stock_info}"
    
    return {
        "query": query,
        "conversation_history_section": conversation_history_section,
        "sec_context_section": sec_context_section,
        "account_info_section": account_info_section,
        "positions_section": positions_section,
        "stock_info_section": stock_info_section
    }


def create_search_query_prompt() -> PromptTemplate:
    """
    Create a prompt template for optimizing search queries.
    
    Returns:
        A PromptTemplate object for generating optimized search queries
    """
    template = """<|system|>
You are a financial search query optimizer. Your job is to generate an optimized search query
for retrieving information from SEC 10-K filings based on the user's query and conversation history.

DO NOT explain your reasoning. ONLY output the optimized search query text.
</|system|>

<|user|>
USER QUERY: {user_query}

CONVERSATION HISTORY:
{conversation_history}

Based on the above, create an optimized search query that:
1. Captures the key financial concepts needed
2. Includes specific financial terms that might be in 10-K documents
3. Extracts and expands any implied searches from the conversation context
4. Uses specific financial terminology that would appear in SEC filings
5. Is focused and precise (between 10-20 words)
</|user|>

<|assistant|>
"""
    
    return PromptTemplate(
        input_variables=["user_query", "conversation_history"],
        template=template
    ) 
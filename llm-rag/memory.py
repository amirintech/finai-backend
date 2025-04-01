"""Memory management for the financial assistant."""

from typing import List, Dict, Optional
import json
import os


class ConversationMemory:
    """
    Manages conversation history for the financial assistant.
    Stores and retrieves conversation history for maintaining context across queries.
    """
    
    def __init__(self, max_history: int = 5, memory_file: Optional[str] = None):
        """
        Initialize the conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to store
            memory_file: Optional path to save conversation history to disk
        """
        self.max_history = max_history
        self.memory_file = memory_file
        self.conversation_history = []
        
        # Load from file if it exists
        if memory_file and os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    self.conversation_history = json.load(f)
                print(f"Loaded conversation history from {memory_file}")
            except Exception as e:
                print(f"Error loading conversation history: {e}")
    
    def add_interaction(self, query: str, response: str) -> None:
        """
        Add a new query-response pair to the conversation history.
        
        Args:
            query: The user's query
            response: The assistant's response
        """
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        # Limit history to max_history items
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Save to file if specified
        if self.memory_file:
            try:
                with open(self.memory_file, 'w') as f:
                    json.dump(self.conversation_history, f, indent=2)
            except Exception as e:
                print(f"Error saving conversation history: {e}")
    
    def get_history_as_text(self) -> str:
        """
        Get conversation history formatted as text for inclusion in prompts.
        
        Returns:
            Formatted conversation history text
        """
        if not self.conversation_history:
            return "No conversation history."
        
        result = []
        for i, interaction in enumerate(self.conversation_history):
            result.append(f"User Query {i+1}: {interaction['query']}")
            result.append(f"Assistant Response {i+1}: {interaction['response']}")
            result.append("")  # Empty line for readability
        
        return "\n".join(result)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
                print(f"Removed conversation history file {self.memory_file}")
            except Exception as e:
                print(f"Error removing conversation history file: {e}") 
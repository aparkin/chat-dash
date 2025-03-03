"""Chat LLM Service implementation.

This module provides the ChatLLM service which acts as a catch-all handler for
messages not processed by other specialized services.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import openai
import re
import textwrap

from .base import ChatService, ServiceResponse, ServiceMessage, MessageType
from .llm_service import LLMServiceMixin

class ChatLLMService(ChatService, LLMServiceMixin):
    """Service for handling general chat interactions with the LLM.
    
    This service acts as the catch-all for messages not handled by other services,
    providing general chat capabilities with context management.
    """
    
    def __init__(self):
        ChatService.__init__(self, "chat_llm")
        LLMServiceMixin.__init__(self, "chat_llm")
        self._service_capabilities_cache = None
        
    def _get_service_capabilities(self) -> str:
        """Get and cache service capabilities from all services.
        
        Returns:
            str: Combined service capabilities text
        """
        if self._service_capabilities_cache is None:
            from . import service_registry
            capabilities = []
            
            for service_name, service in service_registry._services.items():
                if service_name != self.name:  # Skip our own capabilities
                    try:
                        capability = service.get_llm_prompt_addition()
                        if capability and capability.strip():
                            capabilities.append(capability.strip())
                    except Exception as e:
                        print(f"Error getting capabilities for {service_name}: {str(e)}")
            
            self._service_capabilities_cache = "\n\n".join(capabilities)
        
        return self._service_capabilities_cache

    def _process_chat_history(self, chat_history: List[Dict[str, Any]], token_budget: int) -> Tuple[str, str]:
        """Process chat history to extract user intent and learning trajectory.
        
        Args:
            chat_history: List of chat messages
            token_budget: Maximum tokens to use for both summaries combined
            
        Returns:
            Tuple[str, str]: (user intent summary, learning trajectory summary)
        """
        # Allocate token budget: 40% for intent, 60% for trajectory
        intent_budget = int(token_budget * 0.4)
        trajectory_budget = int(token_budget * 0.6)
        
        # Get user messages for intent analysis
        user_messages = []
        for msg in reversed(chat_history):  # Start with most recent
            if msg['role'] == 'user':
                content = msg.get('content', '').strip()
                if content:
                    user_messages.append(content)
                    # Check token count
                    if self.count_tokens("\n".join(user_messages)) > intent_budget:
                        user_messages.pop()  # Remove last message if over budget
                        break
        
        # Get service results and summaries for trajectory
        trajectory_messages = []
        for msg in reversed(chat_history):
            if msg.get('service'):
                msg_type = msg.get('type', '').lower()
                # Include summaries and results without associated summaries
                if msg_type == 'summary' or (
                    msg_type == 'result' and not any(
                        m.get('type', '').lower() == 'summary' 
                        for m in chat_history[chat_history.index(msg):chat_history.index(msg)+2]
                    )
                ):
                    content = msg.get('content', '').strip()
                    if content:
                        trajectory_messages.append({
                            'service': msg.get('service'),
                            'content': content
                        })
                        # Check token count
                        if self.count_tokens(str(trajectory_messages)) > trajectory_budget:
                            trajectory_messages.pop()  # Remove last message if over budget
                            break
        
        # Create prompts for summarization
        intent_prompt = f"""Analyze these user messages and provide a concise summary of what the user is trying to accomplish:

User Messages (from most recent):
{chr(10).join(f"- {msg}" for msg in user_messages)}

Format your response as:
1. Primary Goal: (one sentence)
2. Specific Interests: (bullet points)
3. Current Focus: (what they're working on right now)
"""

        trajectory_prompt = f"""Analyze these service results and provide a concise summary of what has been learned:

Service Results (from most recent):
{chr(10).join(f"[{msg['service']}] {msg['content']}" for msg in trajectory_messages)}

Focus on:
- Key findings and patterns
- Environmental, biological, physical, and chemical insights
- Geographic relationships if relevant
- Any contradictions or unsuccessful approaches
- Most recent findings should be emphasized

Format your response as:
1. Key Findings: (2-3 bullet points)
2. Patterns Identified: (if any)
3. Open Questions: (based on results)
"""

        # Get summaries from LLM
        intent_summary = self._call_llm([{"role": "user", "content": intent_prompt}])
        trajectory_summary = self._call_llm([{"role": "user", "content": trajectory_prompt}])
        
        return intent_summary.strip(), trajectory_summary.strip()
    
    def can_handle(self, message: str) -> bool:
        """Always returns False as this service is called directly when needed.
        
        This service is special and doesn't participate in the normal handler detection.
        It is invoked directly for messages that no other service handles.
        """
        return False
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Package the message for processing."""
        return {
            'message': message,
            'type': 'chat'
        }
    
    def _create_system_message(self, context: Dict[str, Any]) -> str:
        """Create focused system message for chat context.
        
        Creates a comprehensive system message that includes:
        1. Available data sources and their current state
        2. Available service capabilities
        3. User's current goals and progress
        4. Specific interaction guidelines
        """
        # Calculate token budgets (reserving space for final prompt)
        MAX_TOKENS = 8192
        RESERVED_TOKENS = 2000  # For final prompt and response
        
        # Allocate remaining tokens:
        # - 30% for data source info
        # - 20% for service capabilities
        # - 50% for chat history processing
        available_tokens = MAX_TOKENS - RESERVED_TOKENS
        data_token_budget = int(available_tokens * 0.3)
        capabilities_token_budget = int(available_tokens * 0.2)
        history_token_budget = int(available_tokens * 0.5)
        
        # Get dataset information
        datasets = context.get('datasets_store', {})
        dataset_info = []
        dataset_section = []
        if datasets:
            for name, data in datasets.items():
                df_data = data.get('df', [])
                metadata = data.get('metadata', {})
                info = {
                    'name': name,
                    'rows': len(df_data),
                    'columns': metadata.get('columns', []),
                    'selected': name == context.get('selected_dataset')
                }
                dataset_info.append(info)
                
                section = f"- {name}: {info['rows']} rows"
                if info['selected']:
                    section += " (SELECTED)"
                section += f"\n  Columns: {', '.join(info['columns'])}"
                dataset_section.append(section)
                
                # Check token budget
                if self.count_tokens("\n".join(dataset_section)) > data_token_budget // 2:
                    dataset_section.pop()
                    dataset_section.append("... additional datasets omitted for space")
                    break
        
        # Get database information
        db_section = []
        db_structure = context.get('database_structure_store', {})
        if db_structure:
            for table, info in db_structure.items():
                section = f"- {table} ({info['row_count']} rows)"
                section += f"\n  Columns: {', '.join(col['name'] for col in info['columns'])}"
                db_section.append(section)
                
                # Check token budget
                if self.count_tokens("\n".join(db_section)) > data_token_budget // 2:
                    db_section.pop()
                    db_section.append("... additional tables omitted for space")
                    break
        
        # Get service capabilities (cached)
        capabilities = self._get_service_capabilities()
        if self.count_tokens(capabilities) > capabilities_token_budget:
            # If over budget, we'll need to implement a more sophisticated truncation
            # For now, just note that it's truncated
            print("Warning: Service capabilities exceeded token budget")
        
        # Process chat history
        chat_history = context.get('chat_history', [])
        intent_summary, trajectory_summary = self._process_chat_history(
            chat_history, history_token_budget
        )
        
        # Construct final system message with strengthened constraints
        sections = [
            "You are a data analysis assistant focused on helping users understand and work with their data.",
            
            "DOMAIN FOCUS:",
            "1. Scientific and Technical Context:",
            "   - ALWAYS interpret requests in a scientific/technical context first",
            "   - Assume queries about tools, data, or methods refer to their scientific applications",
            "   - When encountering ambiguous terms (e.g., 'monet', 'python'), default to their scientific/technical meanings",
            "   - Only consider artistic, cultural, or general interpretations if explicitly requested",

            "2. Scientific Subject Area Priority:",
            "   - Most scientific topics involve environmental science, biology, chemistry, physics, geography and geology ",
            "   - Organismal taxonomy, physiology, metabolism, genetics, molecular biology, and biochemistry are also a high priority with a focus on omics data, gene annotations and enzyme functions key.",
            "   - When feasible look to link these concepts together and to available data sources.",

            "3. Subject Matter Priority:",
            "   - Primary: Data analysis, scientific methods, research tools relevant to the chat context",
            "   - Secondary: Scientific and technical suggestions for the user to explore given that they are relevant to the chat context",
            "   - Tertiary: General knowledge that supports scientific understanding",
            "   - Avoid: Non-technical interpretations unless specifically requested or can be fairly linked to the chat context",
            
            "AVAILABLE DATA SOURCES:",
            "Datasets:" if dataset_section else "Datasets: None loaded",
            "\n".join(dataset_section) if dataset_section else None,
            "Database Tables:" if db_section else "Database: Not connected",
            "\n".join(db_section) if db_section else None,
            
            "AVAILABLE ACTIONS:",
            capabilities,
            
            "USER CONTEXT:",
            f"Current Goals and Interests:\n{intent_summary}",
            f"Analysis Progress and Findings:\n{trajectory_summary}",
            
            "CRITICAL CONSTRAINTS:",
            "1. ABSOLUTELY NO CODE GENERATION:",
            "   - DO NOT generate any Python code snippets",
            "   - DO NOT suggest modifications to existing code",
            "   - DO NOT provide code examples",
            "   - Instead, use the dataset service's 'analysis:' command for code generation",
            
            "2. ABSOLUTELY NO SQL GENERATION:",
            "   - DO NOT generate any SQL queries",
            "   - DO NOT suggest SQL modifications",
            "   - DO NOT provide SQL examples",
            "   - Instead, tell the user to use the database service's natural language interface",
            
            "3. ABSOLUTELY NO DATA GENERATION:",
            "   - DO NOT create example data tables or datasets",
            "   - DO NOT show hypothetical query results",
            "   - DO NOT display data that hasn't been queried yet",
            
            "4. COMMAND USAGE:",
            "   - ONLY use documented service commands",
            "   - ALWAYS use exact command syntax",
            "   - NEVER invent new commands",
            "   - NEVER modify command syntax",
            
            "INTERACTION RULES:",
            "1. Response Structure:",
            "   - Start with mission understanding",
            "   - Summarize current findings",
            "   - Suggest next steps using ONLY available commands",
            "   - Explain expected insights",
            
            "2. Content Guidelines:",
            "   - Use well formated markdown bullet points for lists",
            "   - Format commands in code blocks",
            "   - Keep responses concise and focused",
            "   - DO NOT repeat context information",
            
            "3. Analysis Flow:",
            "   - Ensure logical operation order",
            "   - Verify prerequisites (e.g., dataset selection)",
            "   - Reference specific IDs when available from the context. Do not make up your own IDs.",
            "   - Build upon previous results",
            
            "4. Error Handling:",
            "   - Identify command syntax errors",
            "   - Suggest correct command usage",
            "   - Explain prerequisites if missing",
            "   - Guide user through error resolution",
            
            "YOUR RESPONSE MUST:",
            "1. Review the current 'mission' (1-2 sentences)",
            "2. Summarize current findings and significance (1-3 sentences)",
            "3. Suggest specific service commands for next steps",
            "4. Explain expected insights from suggested actions",
            "5. Use bullet points for lists",
            "6. Format commands in code blocks",
            "7. NEVER include raw code or SQL",
            "8. NEVER repeat context information",
            '9. Augment the response with relevant general knowledge if appropriate',
            
            "EXAMPLE RESPONSE FORMAT:",
            """
Mission: Analyzing temperature trends in oceanographic data.

Current Findings: Previous analysis revealed seasonal patterns in surface temperatures. Correlation with depth data suggests stratification effects.

Suggested Next Steps:
• Generate detailed temperature analysis:
  ```
  analysis: Analyze temperature variation by depth and season
  ```
• After results, create visualization:
  ```
  analysis: Create heatmap of temperature by depth and season
  ```

Expected Insights:
• Identification of thermocline patterns
• Seasonal mixing dynamics
• Potential impacts on species distribution

General Knowledge:
• The thermocline is the layer of the ocean where the temperature changes rapidly with depth.
• The thermocline is important for the distribution of marine life.
• The thermocline is affected by the seasons and the weather.
"""
        ]
        
        # Join sections, filtering out None values
        return "\n\n".join(section for section in sections if section is not None)
    
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the chat request with proper context management."""
        try:
            # Store context for use in _call_llm - MUST be first
            self.context = context

            # Create system message
            system_message = self._create_system_message(context)
            
            # Get chat history
            chat_history = context.get('chat_history', [])
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # Add relevant history
            # Note: We're starting fresh with history management
            for msg in chat_history[-5:]:  # Start with last 5 messages
                if msg.get('service'):
                    # Format service messages to maintain context
                    messages.append({
                        "role": "assistant",
                        "content": msg['content']
                    })
                else:
                    messages.append({
                        "role": msg.get('role', 'user'),
                        "content": msg.get('content', '')
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": params['message']
            })
            
            # Get LLM response
            response = self._call_llm(messages)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any existing headers
            response = re.sub(r'^(?:Service:[^\n]*\n)?(?:Type:[^\n]*\n\n?)*', '', response, flags=re.IGNORECASE)
            
            # Fix code block wrapping
            def wrap_code_block(match):
                prefix = match.group(1)  # ```language\n
                code = match.group(2)    # code content
                suffix = match.group(3)  # ```
                
                # Split code into lines, wrap each line, and preserve newlines
                wrapped_lines = []
                for line in code.strip().split('\n'):
                    if line.strip():  # Only wrap non-empty lines
                        wrapped_lines.extend(textwrap.wrap(line.strip(), width=80, 
                                                         break_long_words=False, 
                                                         break_on_hyphens=False))
                    else:
                        wrapped_lines.append('')  # Preserve empty lines
                
                # Reconstruct code block
                wrapped_code = '\n'.join(wrapped_lines)
                
                return f"{prefix}{wrapped_code}\n{suffix}"
            
            # Apply code block wrapping
            response = re.sub(
                r'(```(?:[^\n]*)\n)(.*?)(```)',
                wrap_code_block,
                response,
                flags=re.DOTALL
            )
            
            # Return formatted response with service header
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=response,  # Just the response content, no headers
                        message_type=MessageType.RESULT,
                        role="assistant"
                    )
                ]
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error processing chat message: {str(e)}",
                        message_type=MessageType.ERROR,
                        role="assistant"
                    )
                ]
            )
    
    def get_help_text(self) -> str:
        """Get help text for chat capabilities."""
        return """
💭 **General Chat**
- Ask questions about your data
- Get analysis suggestions
- Request explanations
- Follow up on previous results
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for chat capabilities."""
        return """
Chat Capabilities:
- Answer questions about available data
- Suggest analysis approaches
- Explain results and findings
- Maintain conversation context
- Integrate with other services
"""

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process a message using the LLM.
        
        This service doesn't use process_message directly as it handles
        messages through the execute method with full context.
        """
        raise NotImplementedError("ChatLLMService uses execute method instead of process_message")

    def summarize(self, content: str, chat_history: List[Dict[str, Any]]) -> str:
        """Summarize content using the LLM.
        
        This service doesn't use summarize directly as it provides
        direct chat responses through execute.
        """
        raise NotImplementedError("ChatLLMService uses execute method instead of summarize")
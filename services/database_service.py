"""
Database service implementation.

This service handles SQL query execution and database interactions in the ChatDash application.
It provides a modular interface for:
1. SQL query detection and validation
2. Safe query execution
3. Result formatting and storage
4. Database state management
"""

from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime
import pandas as pd
import sqlite3
from pathlib import Path
import traceback
import json
from enum import Enum, auto
import numpy as np

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    PreviewIdentifier,
    MessageType
)
from .llm_service import LLMServiceMixin

class RequestType(Enum):
    SQL_EXECUTION = auto()
    QUERY_SEARCH = auto()
    DATABASE_INFO = auto()
    SERVICE_TEST = auto()
    NATURAL_QUERY = auto()
    QUERY_EXPLAIN = auto()
    DATASET_CONVERSION = auto()

class DatabaseService(ChatService, LLMServiceMixin):
    """Service for database query handling and execution."""
    
    def __init__(self):
        ChatService.__init__(self, "database")
        LLMServiceMixin.__init__(self, "database")
        # Register our prefix for query IDs
        PreviewIdentifier.register_prefix("query")
        
        # SQL code block pattern
        self.sql_block_pattern = r'```sql\s*(.*?)```'
        
        # Query execution patterns
        self.execution_patterns = [
            # Direct execution commands
            r'^(?:search|query)\s+query_\d{8}_\d{6}(?:_orig|_alt\d+)\b',  # Handle search/query for query IDs
            r'^(?:search|query)[.!]$',  # Simple execution commands
            
            # Database info request
            r'tell\s+me\s+about\s+my\s+database\b',
            
            # Dataset conversion
            r'^convert\s+query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b',
            
            # Self-test command
            r'^test\s+database\s+service\b'
        ]
        
        # Compile patterns for efficiency
        self.sql_block_re = re.compile(self.sql_block_pattern, re.IGNORECASE | re.DOTALL)
        self.execution_res = [re.compile(p, re.IGNORECASE) for p in self.execution_patterns]
        
        # Write operations that are not allowed
        self.write_operations = {'insert', 'update', 'delete', 'drop', 'alter', 'create'}
    
    def can_handle(self, message: str) -> bool:
        """Detect if message contains SQL or database commands.
        
        Handles:
        1. SQL code blocks (```sql SELECT * FROM table```)
        2. Query search commands (search query_20240101_123456)
        3. Database info requests (tell me about my database)
        4. Self-test command (test database service)
        5. Natural language database commands (database: ...)
        6. Query explanation requests (explain query_ID)
        """
        # Clean and normalize message
        message = message.strip()
        message_lower = message.lower()
        
        # Check for SQL code blocks
        if self.sql_block_re.search(message):
            return True
            
        # Check for execution commands and test command
        for pattern in self.execution_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
                
        # Check for database: command
        if message_lower.startswith("database:"):
            return True
        
        # Check for explain command with valid query ID
        if message_lower.startswith("explain "):
            # Get query IDs (handling potential comma-separated list)
            remainder = message_lower[8:].strip()
            if not remainder:
                return False
            
            query_ids = [id.strip() for id in remainder.split(',')]
            return all(
                re.match(r'^query_\d{8}_\d{6}(?:_orig|_alt\d+)$', query_id)
                for query_id in query_ids
            )
        
        return False
    
    def parse_request(self, message: str) -> Tuple[RequestType, Dict[str, Any]]:
        """Parse message to determine request type and extract relevant parameters.
        
        Returns:
            Tuple[RequestType, Dict[str, Any]]: Request type and parameters dict
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Check for SQL code blocks
        if match := self.sql_block_re.search(message):
            return RequestType.SQL_EXECUTION, {"sql": match.group(1).strip()}
        
        # Check for simple query search
        if message_lower in ['search.', 'query.']:
            return RequestType.QUERY_SEARCH, {"query_id": None}  # Will use most recent query
        
        # Check for specific query search
        if match := re.match(r'^search\s+query_\d{8}_\d{6}(?:_orig|_alt\d+)$', message_lower):
            return RequestType.QUERY_SEARCH, {"query_id": match.group(0)[7:]}
        
        # Check for database info request
        if message_lower == 'tell me about my database':
            return RequestType.DATABASE_INFO, {}
        
        # Check for test command
        if message_lower == "test database service":
            return RequestType.SERVICE_TEST, {}
        
        # Check for database: command
        if message_lower.startswith("database:"):
            return RequestType.NATURAL_QUERY, {"query": message[9:].strip()}
        
        # Check for dataset conversion
        if match := re.match(r'^convert\s+(query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message_lower):
            return RequestType.DATASET_CONVERSION, {"query_id": match.group(1)}
        
        # Check for explain command
        if message_lower.startswith("explain "):
            remainder = message_lower[8:].strip()
            query_ids = [id.strip() for id in remainder.split(',')]
            if all(re.match(r'^query_\d{8}_\d{6}(?:_orig|_alt\d+)$', query_id) for query_id in query_ids):
                return RequestType.QUERY_EXPLAIN, {"query_ids": query_ids}
            
        raise ValueError(f"Unable to parse request from message: {message}")

    def find_recent_query(self, chat_history: list, query_id: str = None) -> tuple[str, str]:
        """Find SQL query in chat history.
        
        Looks for SQL blocks in both assistant messages and service messages.
        """
        print(f"\nSearching for query in chat history...")
        print(f"Target query ID: {query_id}")
        
        for msg in reversed(chat_history):
            # Check both assistant messages and database service messages
            if ('```sql' in msg['content'].lower() and 
                (msg['role'] == 'assistant' or 
                 (msg.get('service') == self.name and msg['role'] == 'system'))):
                content = msg['content']
                print(f"\nFound SQL block in message:")
                print(f"Message role: {msg['role']}")
                print(f"Message service: {msg.get('service')}")
                
                # Extract all SQL blocks with IDs
                sql_blocks = []
                for match in re.finditer(r'```sql\s*(.*?)```', content, re.DOTALL):
                    block = match.group(1).strip()
                    #print(f"\nRaw SQL block:\n{block}")
                    
                    id_match = re.search(r'--\s*Query ID:\s*(query_\d{8}_\d{6}(?:_orig|_alt\d+))\b', block)
                    if id_match:
                        found_id = id_match.group(1)
                        # Remove ID comment from query
                        query = re.sub(r'--\s*Query ID:.*?\n', '', block).strip()
                        print(f"\nFound query with ID {found_id}:")
                        #print(f"Query text:\n{query}")
                        sql_blocks.append((query, found_id))
                
                if sql_blocks:
                    if query_id:
                        # Find specific query
                        for query, found_id in sql_blocks:
                            if found_id == query_id:
                                print(f"\nFound requested query: {query_id}")
                                print(f"Query text to execute:\n{query}")
                                return query, found_id
                    else:
                        # Find most recent original query
                        for query, found_id in sql_blocks:
                            if found_id.endswith('_orig'):
                                print(f"\nFound most recent original query: {found_id}")
                                print(f"Query text to execute:\n{query}")
                                return query, found_id
        
        print("No matching query found")
        return None, None
    
    def _validate_sql_query(self, sql: str, db_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate SQL query for safety and correctness.
        
        Args:
            sql: SQL query to validate
            db_path: Path to SQLite database
            
        Returns:
            Tuple[bool, str, dict]: (is_valid, error_message, metadata)
                - is_valid: True if query is safe and valid
                - error_message: Description of any issues found
                - metadata: Additional information about the query
        """
        try:
            # 1. Basic safety checks
            
            # Remove SQL comments before checking for write operations
            sql_no_comments = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove single line comments
            sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)  # Remove multi-line comments
            sql_lower = sql_no_comments.lower().strip()
            
            # Check for write operations
            for op in self.write_operations:
                if sql_lower.startswith(op) or f' {op} ' in sql_lower:
                    return False, f"Write operation '{op}' is not allowed", {}
            
            # 2. Connect to database for deeper validation
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 3. Get schema information
            tables = {}
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                cursor.execute(f"PRAGMA table_info({table_name})")
                tables[table_name] = {row[1]: row[2] for row in cursor.fetchall()}
            
            # 4. Explain query plan to validate syntax and references
            try:
                cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
                plan = cursor.fetchall()
                
                # Extract referenced tables from plan
                referenced_tables = set()
                for row in plan:
                    plan_detail = row[3].lower()
                    for table in tables.keys():
                        if table.lower() in plan_detail:
                            referenced_tables.add(table)
                
                metadata = {
                    'referenced_tables': list(referenced_tables),
                    'schema': {t: list(cols.keys()) for t, cols in tables.items()},
                    'plan': plan
                }
                
                return True, "", metadata
                
            except sqlite3.Error as e:
                return False, f"SQL syntax error: {str(e)}", {}
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", {}
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _execute_sql_query(self, query: str, db_path: str) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """Execute SQL query and return results with metadata.
        
        Args:
            query: SQL query to execute
            db_path: Path to SQLite database
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], str]: (results, metadata, preview)
                - results: Query results as DataFrame
                - metadata: Query execution metadata
                - preview: Formatted preview of results
        """
        try:
            # First validate the query
            is_valid, error_msg, validation_metadata = self._validate_sql_query(query, db_path)
            if not is_valid:
                raise Exception(error_msg)
            
            # Execute query
            conn = sqlite3.connect(db_path)
            
            # First, explain the query
            explain = pd.read_sql(f"EXPLAIN QUERY PLAN {query}", conn)
            plan = "\n".join(explain.to_string().split('\n'))
            
            # Then execute it
            results = pd.read_sql(query, conn)
            
            # Format preview
            preview = "\n\n```\n" + results.head().to_string() + "\n```\n\n"
            
            metadata = {
                'rows': len(results),
                'columns': list(results.columns),
                'execution_plan': plan,
                **validation_metadata  # Include validation metadata
            }
            
            return results, metadata, preview
            
        except sqlite3.Error as e:
            # Pass through SQL errors directly
            raise sqlite3.Error(str(e))
        except Exception as e:
            # Pass through other errors
            raise
            
        finally:
            if 'conn' in locals():
                conn.close()
    
    def execute(self, request: Tuple[RequestType, Dict[str, Any]], context: dict) -> ServiceResponse:
        """Execute database service request.
        
        Args:
            request: Tuple of (RequestType, params) from parse_request
            context: Execution context with:
                - chat_history: List of chat messages
                - database_state: Current database connection info
                - database_structure: Database schema info
                
        Returns:
            ServiceResponse with:
            - messages: List of service messages
            - context: Updated context
            - store_updates: Updates to query store
        """
        try:
            # Store context for use in _call_llm
            self.context = context
            # Check database connection
            if not context.get('database_state', {}).get('connected'):
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No database connected. Please connect to a database first.",
                        message_type=MessageType.ERROR
                    )]
                )
            
            db_path = context['database_state']['path']
            request_type, request_params = request
            
            if request_type == RequestType.SQL_EXECUTION:
                # Execute SQL query
                query_text = request_params.get('sql')
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No SQL query provided.",
                            message_type=MessageType.ERROR
                        )]
                    )
                
                # Execute query
                try:
                    results, metadata, preview = self._execute_sql_query(query_text, db_path)
                    
                    # TODO: We may need to do column type coercion here. 
                    
                    # Generate query ID
                    query_id = PreviewIdentifier.create_id(prefix="query")
                    
                    # Store successful query
                    store_updates = {
                        'successful_queries_store': {
                            query_id: {
                                'sql': query_text,
                                'metadata': metadata,
                                'execution_time': datetime.now().isoformat(),
                                'dataframe': results.to_dict('records')  # Store the results
                            }
                        }
                    }
                    
                    # Format response
                    response = f"""Query executed successfully!

Results preview:

Query ID: {query_id}
{preview}

Total rows: {metadata['rows']}

Execution plan:
{metadata['execution_plan']}

To convert your result to a dataset you can use 'convert {query_id} to dataset'"""
                    
                    # Create messages list with preview
                    messages = [
                        ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )
                    ]
                    
                    # Generate LLM summary if we have results
                    if not results.empty:
                        try:
                            llm_summary = self.summarize(results, context['chat_history'], context)
                            if llm_summary:
                                messages.append(
                                    ServiceMessage(
                                        service=self.name,
                                        content=f"\n### Analysis Summary\n\n{llm_summary}",
                                        message_type=MessageType.SUMMARY
                                    )
                                )
                        except Exception as e:
                            print(f"Error generating LLM summary: {str(e)}")
                            # Continue without summary
                    
                    return ServiceResponse(
                        messages=messages,
                        store_updates=store_updates,
                        state_updates={'chat_input': ''}
                    )
                    
                except sqlite3.Error as e:
                    # Handle SQL errors
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"SQL Error: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                except Exception as e:
                    # Handle other errors
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.QUERY_SEARCH:
                # Find query to execute
                query_text, query_id = self.find_recent_query(
                    context['chat_history'],
                    request_params.get('query_id')
                )
                
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No SQL query found in recent chat history. Please make sure a query has been suggested before using 'search.' or 'query.'",
                            message_type=MessageType.ERROR
                        )]
                    )
                
                # Execute found query
                try:
                    results, metadata, preview = self._execute_sql_query(query_text, db_path)
                    
                    # Store successful query
                    store_updates = {
                        'successful_queries_store': {
                            query_id: {
                                'sql': query_text,
                                'metadata': metadata,
                                'execution_time': datetime.now().isoformat(),
                                'dataframe': results.to_dict('records')
                            }
                        }
                    }
                    
                    # Format response
                    response = f"""Query executed successfully!

Results preview:

Query ID: {query_id}
{preview}

Total rows: {metadata['rows']}

Execution plan:
{metadata['execution_plan']}

To convert your result to a dataset you can use 'convert {query_id} to dataset'"""
                    
                    messages = [
                        ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )
                    ]
                    
                    # Generate LLM summary if we have results
                    if not results.empty:
                        try:
                            llm_summary = self.summarize(results, context['chat_history'], context)
                            if llm_summary:
                                messages.append(
                                    ServiceMessage(
                                        service=self.name,
                                        content=f"\n### Analysis Summary\n\n{llm_summary}",
                                        message_type=MessageType.SUMMARY
                                    )
                                )
                        except Exception as e:
                            print(f"Error generating LLM summary: {str(e)}")
                            # Continue without summary
                    
                    return ServiceResponse(
                        messages=messages,
                        store_updates=store_updates,
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error executing query: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.DATABASE_INFO:
                # Get database structure from context
                structure = context.get('database_structure', {})
                if not structure:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="Database structure information not available.",
                            message_type=MessageType.ERROR
                        )]
                    )
                
                # Format database overview
                overview = ["### Database Overview"]
                
                # Add table summaries
                for table, info in structure.items():
                    overview.append(f"\n**{table}** ({info['row_count']} rows)")
                    
                    # Add column details
                    overview.append("\nColumns:")
                    for col in info['columns']:
                        constraints = []
                        if col['pk']: constraints.append("PRIMARY KEY")
                        if col['notnull']: constraints.append("NOT NULL")
                        if col['default']: constraints.append(f"DEFAULT {col['default']}")
                        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                        overview.append(f"- {col['name']}: {col['type']}{constraint_str}")
                    
                    # Add foreign key relationships
                    if info['foreign_keys']:
                        overview.append("\nForeign Keys:")
                        for fk in info['foreign_keys']:
                            overview.append(f"- {fk['from']} → {fk['table']}.{fk['to']}")
                    
                    overview.append("")  # Add spacing between tables
                
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="\n".join(overview),
                        message_type=MessageType.INFO
                    )],
                    state_updates={'chat_input': ''}
                )
            
            elif request_type == RequestType.SERVICE_TEST:
                return self._run_self_test(context)
            
            elif request_type == RequestType.NATURAL_QUERY:
                # Get query text
                query_text = request_params.get('query')
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No query text provided.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Get database structure
                db_structure = context.get('database_structure', {})
                if not db_structure:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No database structure available. Please ensure a database is connected.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Generate prompt
                prompt = self._create_natural_query_prompt(query_text, db_structure, context)
                
                try:
                    # Get LLM response
                    response = self._call_llm([{"role": "user", "content": prompt}])
                    
                    # Process response to ensure proper query ID formatting
                    response = self.add_ids_to_blocks(response)
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )],
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error generating SQL queries: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.QUERY_EXPLAIN:
                # Get query IDs to explain
                query_ids = request_params.get('query_ids', [])
                focus = request_params.get('focus', '')
                
                if not query_ids:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No query IDs provided to explain.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Get database structure
                db_structure = context.get('database_structure', {})
                if not db_structure:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No database structure available. Please ensure a database is connected.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Get query details
                queries = []
                missing_queries = []
                for query_id in query_ids:
                    query_details = self._get_query_details(query_id, context)
                    if query_details:
                        queries.append(query_details)
                    else:
                        missing_queries.append(query_id)
                
                if missing_queries:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Could not find the following queries: {', '.join(missing_queries)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Generate prompt
                prompt = self._create_explain_prompt(queries, db_structure, focus)
                
                try:
                    # Get LLM analysis
                    analysis = self._call_llm([{"role": "user", "content": prompt}])
                    
                    # Process response to ensure proper query ID formatting
                    analysis = self.add_ids_to_blocks(analysis)
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=analysis,
                            message_type=MessageType.RESULT
                        )],
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error generating query explanation: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.DATASET_CONVERSION:
                return self._handle_dataset_conversion(request_params, context)
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Invalid request type",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}  # Clear input even on error
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Service error: {str(e)}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}  # Clear input on error
            )
    
    def _run_self_test(self, context: Dict[str, Any]) -> ServiceResponse:
        """Run self-tests for database service functionality.
        
        This method tests core functionality within the current application context:
        1. SQL parsing and validation
        2. Query execution (if database connected)
        3. Error handling
        4. Context management
        """
        test_results = []
        passed = 0
        total = 0
        
        def run_test(name: str, test_fn) -> bool:
            nonlocal passed, total
            total += 1
            try:
                test_fn()
                test_results.append(f"✓ {name}")
                passed += 1
                return True
            except Exception as e:
                test_results.append(f"✗ {name}: {str(e)}")
                return False
        
        # 1. Test SQL parsing
        def test_sql_parsing():
            message = "```sql\nSELECT * FROM table\n```"
            request_type, params = self.parse_request(message)
            assert request_type == RequestType.SQL_EXECUTION
            assert params['sql'] == 'SELECT * FROM table'
        
        run_test("SQL parsing", test_sql_parsing)
        
        # 2. Test command parsing
        def test_command_parsing():
            message = "test database service"
            request_type, params = self.parse_request(message)
            assert request_type == RequestType.SERVICE_TEST
            assert params == {}
        
        run_test("Command parsing", test_command_parsing)
        
        # 3. Test query validation
        def test_query_validation():
            # Test write operation detection
            is_valid, error, _ = self._validate_sql_query(
                "INSERT INTO table VALUES (1)",
                context['database_state']['path']
            )
            assert not is_valid
            assert "not allowed" in error.lower()
            
            # Test valid query
            is_valid, error, metadata = self._validate_sql_query(
                "SELECT 1",
                context['database_state']['path']
            )
            assert is_valid
            assert not error
        
        if context.get('database_state', {}).get('connected'):
            run_test("Query validation", test_query_validation)
            
            # 4. Test query execution
            def test_query_execution():
                df, metadata, preview = self._execute_sql_query(
                    "SELECT 1 as test",
                    context['database_state']['path']
                )
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 1
                assert 'test' in df.columns
            
            run_test("Query execution", test_query_execution)
        else:
            test_results.append("⚠ Skipped database tests - no connection")
        
        # 5. Test error handling
        def test_error_handling():
            # Test invalid SQL parsing
            response = self.execute(
                (RequestType.SQL_EXECUTION, {'sql': 'INVALID SQL'}),
                {'database_state': {'connected': True, 'path': ':memory:'}}
            )
            assert response.messages[0].message_type == MessageType.ERROR
            assert "error" in response.messages[0].content.lower()
        
        run_test("Error handling", test_error_handling)
        
        # Format results
        summary = [
            "### Database Service Self-Test Results",
            f"\nPassed: {passed}/{total} tests\n",
            "Detailed Results:"
        ] + test_results
        
        if context.get('database_state', {}).get('connected'):
            summary.append("\nTests run with active database connection")
        else:
            summary.append("\n⚠ Some tests skipped - no database connection")
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content="\n".join(summary),
                message_type=MessageType.INFO if passed == total else MessageType.WARNING
            )]
        )

    def detect_content_blocks(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect SQL code blocks in text.
        
        Returns:
            List of tuples (sql_content, start_pos, end_pos)
        """
        blocks = []
        for match in self.sql_block_re.finditer(text):
            start, end = match.span()
            content = match.group(1).strip()
            blocks.append((content, start, end))
        return blocks
        
    def add_ids_to_blocks(self, text: str) -> str:
        """Add query IDs to SQL blocks in text.
        
        Adds a Query ID comment at the start of each SQL block while preserving existing comments:
        ```sql
        -- Query ID: query_YYYYMMDD_HHMMSS_[orig|altN]
        -- Purpose: Clear description of query goal
        -- Tables: List of tables used
        -- Assumptions: Any important assumptions
        SELECT ...
        ```
        """
        if '```sql' not in text.lower():
            return text
            
        modified_response = []
        current_pos = 0
        
        # Get base ID for this response
        base_id = PreviewIdentifier.create_id(prefix="query")
        previous_id = base_id  # Track the last generated ID
        block_counter = 0
        
        # Process each SQL block
        for sql_content, start, end in self.detect_content_blocks(text):
            # Add text before this block
            modified_response.append(text[current_pos:start])
            
            # Skip if block already has a Query ID
            if re.search(r'^--\s*Query ID:', sql_content):
                modified_response.append(text[start:end])
            else:
                # Generate new ID based on counter
                if block_counter == 0:
                    query_id = base_id
                else:
                    # Use the previous ID to generate the next alternative
                    query_id = PreviewIdentifier.create_id(previous_id=previous_id)
                
                previous_id = query_id  # Update previous_id for next iteration
                block_counter += 1
                
                # Split content into lines
                lines = sql_content.split('\n')
                
                # Find where comments end and SQL begins
                sql_start = 0
                for i, line in enumerate(lines):
                    if not line.strip().startswith('--'):
                        sql_start = i
                        break
                
                # Reconstruct the block with ID
                comment_lines = lines[:sql_start]
                sql_lines = lines[sql_start:]
                
                # Add Query ID as first comment
                comment_lines.insert(0, f"-- Query ID: {query_id}")
                
                # Combine everything back
                modified_content = '\n'.join(comment_lines + sql_lines)
                
                # Add ID comment at the start of the block
                block_parts = text[start:end].split(sql_content)
                modified_block = f"{block_parts[0]}{modified_content}{block_parts[1]}"
                modified_response.append(modified_block)
            
            current_pos = end
        
        # Add any remaining text
        modified_response.append(text[current_pos:])
        
        return ''.join(modified_response)

    def _handle_dataset_conversion(self, params: dict, context: dict) -> ServiceResponse:
        """Handle conversion of query results to dataset.
        
        Args:
            params: Parameters including query_id
            context: Execution context with query store
            
        Returns:
            ServiceResponse with conversion result
        """
        query_id = params.get('query_id')
        if not query_id:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No query ID provided for conversion.",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get stored execution
        executions = context.get('successful_queries_store', {})
        stored = executions.get(query_id)
        
        if not stored:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"No execution found for query ID: {query_id}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
        
        try:
            # Re-execute query to get fresh results
            db_path = context['database_state']['path']
            results, metadata, _ = self._execute_sql_query(stored['sql'], db_path)
            
            # Process DataFrame for consistency with uploaded datasets
            try:
                # Define common missing value indicators
                missing_values = [
                    '-', 'NA', 'na', 'N/A', 'n/a',
                    'NaN', 'nan', 'NAN',
                    'None', 'none', 'NONE',
                    'NULL', 'null', 'Null',
                    'ND', 'nd', 'N/D', 'n/d',
                    '', ' '  # Empty strings and spaces
                ]
                
                # Track transformations for metadata
                transformations = []
                
                # Replace missing values
                results = results.replace(missing_values, np.nan)
                if results.isna().any().any():
                    transformations.append("Standardized missing values")
                
                # Attempt numeric conversion for string columns
                numeric_conversions = []
                for col in results.select_dtypes(include=['object']).columns:
                    try:
                        # Check if column contains only numeric values (allowing NaN)
                        non_nan = results[col].dropna()
                        if len(non_nan) > 0 and non_nan.astype(str).str.match(r'^-?\d*\.?\d+$').all():
                            results[col] = pd.to_numeric(results[col], errors='coerce')
                            numeric_conversions.append(col)
                    except Exception:
                        continue
                
                if numeric_conversions:
                    transformations.append(f"Converted columns to numeric: {', '.join(numeric_conversions)}")
                
                # Clean column names
                old_columns = list(results.columns)
                results.columns = results.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
                if any(old != new for old, new in zip(old_columns, results.columns)):
                    transformations.append("Cleaned column names")
                
            except Exception as e:
                print(f"Warning: Error during DataFrame processing: {str(e)}")
                transformations.append(f"Note: Some data cleaning failed: {str(e)}")
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
            # Generate profile report
            try:
                from pandas_profiling import ProfileReport
                profile = ProfileReport(
                    results,
                    minimal=True,
                    title=f"Profile Report for {query_id}",
                    html={'style': {'full_width': True}},
                    progress_bar=False,
                    correlations={'pearson': {'calculate': True}},
                    missing_diagrams={'matrix': False},
                    samples=None
                )
                profile_html = profile.to_html()
            except Exception as e:
                print(f"Warning: Profile report generation failed: {str(e)}")
                profile_html = None
            
            # Create dataset with special metadata
            datasets[query_id] = {
                'df': results.to_dict('records'),
                'metadata': {
                    'source': f"SQL Query: {query_id}",
                    'sql_query': stored['sql'],
                    'execution_plan': metadata['execution_plan'],
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(results),
                    'columns': list(results.columns),
                    'referenced_tables': metadata.get('referenced_tables', []),
                    'transformations': transformations  # Add transformation history
                },
                'profile_report': profile_html
            }
            
            # Format transformation message
            transform_msg = "\n- " + "\n- ".join(transformations) if transformations else ""
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Query results converted to dataset '{query_id}'

- Rows: {len(results)}
- Columns: {', '.join(results.columns)}
- Source: SQL Query {query_id}
- Referenced tables: {', '.join(metadata.get('referenced_tables', []))}
Data Transformations:{transform_msg}""",
                    message_type=MessageType.INFO
                )],
                store_updates={
                    'datasets_store': datasets,  # Update datasets store
                    'successful_queries_store': {  # Update queries store, removing the converted query
                        k: v for k, v in executions.items() if k != query_id
                    }
                },
                state_updates={'chat_input': ''}
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Database service error: ❌ Query result conversion failed: {str(e)}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )

    def summarize(self, df: pd.DataFrame, chat_history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate comprehensive summary of database query results.
        
        Args:
            df: DataFrame containing query results
            chat_history: List of previous chat messages
            context: Execution context with:
                - database_structure: Schema information
                - database_state: Connection info
                - successful_queries_store: Query history with metadata
        """
        try:
            # 1. Validate database state and structure
            if not context.get('database_state', {}).get('connected'):
                return "No database connected. Cannot generate summary."
                
            structure = context.get('database_structure', {})
            if not structure:
                return "Database structure information not available for analysis."
                
            # 2. Get query information from successful_queries_store
            queries_store = context.get('successful_queries_store', {})
            current_query = None
            query_metadata = None
            
            # Find the most recent query that matches our results
            for query_id, stored in queries_store.items():
                stored_df = pd.DataFrame(stored['dataframe'])
                if len(stored_df) == len(df) and all(stored_df.columns == df.columns):
                    current_query = stored['sql']
                    query_metadata = stored['metadata']
                    break
            
            # 3. Create base system prompt
            system_prompt = self._create_analysis_prompt(
                structure=structure,
                current_query=current_query,
                query_metadata=query_metadata
            )
            
            # 4. Calculate token budgets
            prompt_tokens = self.count_tokens(system_prompt)
            available_tokens = self._calculate_result_token_budget(prompt_tokens)
            
            # 5. Package results within token limit
            result_content = self._package_results_for_analysis(
                df=df,
                available_tokens=available_tokens,
                structure=structure,
                query_metadata=query_metadata
            )
            
            # 6. Get LLM response
            response = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": result_content}
            ])
            
            return response.strip()
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating summary: {str(e)}"

    def _create_analysis_prompt(self, structure: Dict, current_query: Optional[str], query_metadata: Optional[Dict]) -> str:
        """Create analysis prompt with database context and query information."""
        prompt = """You are a database analysis expert. Analyze these SQL query results in the context of:

1. Database Structure:
{db_structure}

2. Executed Query:
{query_info}

Provide analysis focusing on:
1. Data Content:
   - Key patterns and relationships
   - Statistical insights
   - Data quality observations
   - Correlations between columns

2. Domain-Specific Analysis:
   - Taxonomic patterns
   - Gene functions and pathways
   - Metabolic/physiological insights
   - Environmental relationships
   - Geographic/geological context
   - Soil/water characteristics

3. Query Assessment:
   - Coverage of referenced tables
   - Effectiveness of joins and filters
   - Data completeness
   - Performance implications

4. Recommendations:
   - Additional analyses
   - Query optimizations
   - Related data to consider

Focus on providing concrete insights based on the actual data while considering the database structure and query context."""

        # Format database structure section
        db_structure = []
        referenced_tables = query_metadata.get('referenced_tables', []) if query_metadata else []
        
        for table, info in structure.items():
            if table in referenced_tables:
                db_structure.append(f"\n{table} (Referenced):")
            else:
                db_structure.append(f"\n{table}:")
            for col in info['columns']:
                constraints = []
                if col['pk']: constraints.append("PRIMARY KEY")
                if col['notnull']: constraints.append("NOT NULL")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                db_structure.append(f"- {col['name']}: {col['type']}{constraint_str}")

        # Format query information
        query_info = []
        if current_query:
            query_info.append(f"SQL Query:\n{current_query}")
        if query_metadata:
            if 'execution_plan' in query_metadata:
                query_info.append(f"\nExecution Plan:\n{query_metadata['execution_plan']}")
            if 'referenced_tables' in query_metadata:
                query_info.append(f"\nReferenced Tables: {', '.join(query_metadata['referenced_tables'])}")

        return prompt.format(
            db_structure="\n".join(db_structure),
            query_info="\n".join(query_info) if query_info else "Query information not available"
        )

    def _calculate_result_token_budget(self, prompt_tokens: int) -> int:
        """Calculate available tokens for result content."""
        # Get model's maximum context length
        max_tokens = 8192  # We could make this configurable per model
        
        # Reserve tokens for:
        # - System prompt (already counted in prompt_tokens)
        # - Response generation (1500 tokens)
        # - Safety margin (500 tokens)
        reserved_tokens = prompt_tokens + 2000
        
        # Return available tokens for results
        return max_tokens - reserved_tokens

    def _package_results_for_analysis(self, df: pd.DataFrame, available_tokens: int, structure: Dict, query_metadata: Optional[Dict]) -> str:
        """Package query results for analysis within token limit.
        
        This method creates a structured representation of the query results
        that fits within the available token budget while preserving the most
        important analytical information.
        
        Args:
            df: DataFrame containing query results
            available_tokens: Maximum tokens available for results
            structure: Database structure information
            query_metadata: Query execution metadata
            
        Returns:
            str: JSON-formatted string containing packaged results
        """
        try:
            # Start with basic result information
            content = {
                'summary': {
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.astype(str).to_dict()
                }
            }
            
            # Add numeric column statistics
            content['numeric_analysis'] = {}
            for col in df.select_dtypes(include=['number']).columns:
                stats = df[col].describe()
                content['numeric_analysis'][col] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'quartiles': {
                        '25%': stats['25%'],
                        '50%': stats['50%'],
                        '75%': stats['75%']
                    }
                }
            
            # Add categorical column analysis
            content['categorical_analysis'] = {}
            for col in df.select_dtypes(exclude=['number']).columns:
                value_counts = df[col].value_counts()
                # Only include top categories if there are many
                if len(value_counts) <= 10:
                    content['categorical_analysis'][col] = {
                        'unique_values': len(value_counts),
                        'top_values': value_counts.to_dict()
                    }
                else:
                    content['categorical_analysis'][col] = {
                        'unique_values': len(value_counts),
                        'top_values': value_counts.head(10).to_dict(),
                        'other_categories': len(value_counts) - 10
                    }
            
            # Add null value analysis
            null_counts = df.isnull().sum()
            if null_counts.any():
                content['null_analysis'] = null_counts.to_dict()
            
            # Convert initial content to check tokens
            initial_content = json.dumps(content, indent=2)
            initial_tokens = self.count_tokens(initial_content)
            
            # If we have room for samples, add them progressively
            if initial_tokens < available_tokens:
                remaining_tokens = available_tokens - initial_tokens
                samples = self._get_progressive_samples(df, remaining_tokens)
                if samples:
                    content['samples'] = samples
            
            # Final formatting with indentation for readability
            return json.dumps(content, indent=2)
            
        except Exception as e:
            print(f"Error packaging results: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return basic information on error
            return json.dumps({
                'error': str(e),
                'basic_info': {
                    'total_rows': len(df),
                    'columns': list(df.columns)
                }
            })
    
    def _get_progressive_samples(self, df: pd.DataFrame, available_tokens: int) -> Optional[List[Dict]]:
        """Get as many sample rows as possible within token limit.
        
        Uses a progressive sampling strategy:
        1. Start with a small sample
        2. Gradually add more samples if space permits
        3. Ensure samples are representative
        
        Args:
            df: DataFrame to sample from
            available_tokens: Maximum tokens available for samples
            
        Returns:
            List of sample records or None if no space for samples
        """
        try:
            samples = []
            base_sample_size = min(5, len(df))
            
            while True:
                # Get next batch of samples
                new_samples = df.sample(n=base_sample_size).to_dict('records')
                temp_samples = samples + new_samples
                
                # Check if adding these samples would exceed token budget
                if self.count_tokens(json.dumps(temp_samples)) > available_tokens:
                    break
                
                samples = temp_samples
                if len(samples) >= len(df):
                    break
                
                # Increase sample size for next iteration
                base_sample_size = min(base_sample_size * 2, len(df) - len(samples))
            
            return samples if samples else None
            
        except Exception as e:
            print(f"Error getting samples: {str(e)}")
            return None

    def _replace_query_references(self, message: str, context: dict) -> str:
        """Replace query IDs in message with their SQL and metadata.
        
        Args:
            message: Message containing query IDs
            context: Execution context with successful_queries_store and chat_history
            
        Returns:
            Message with query IDs replaced with SQL and metadata
        """
        # Get successful queries store
        queries_store = context.get('successful_queries_store', {})
        
        # Find all query IDs in message
        query_pattern = r'query_\d{8}_\d{6}(?:_orig|_alt\d+)'
        
        def replace_query(match):
            query_id = match.group(0)
            
            # First try successful queries store
            stored = queries_store.get(query_id)
            if stored:
                return f"""Query {query_id}:
```sql
{stored['sql']}
```
Execution Time: {stored['execution_time']}
Rows: {stored['metadata']['rows']}
Tables: {', '.join(stored['metadata'].get('referenced_tables', []))}
"""
            
            # Fall back to chat history
            query_text, found_id = self.find_recent_query(context.get('chat_history', []), query_id)
            if query_text:
                return f"""Query {query_id}:
```sql
{query_text}
```
Note: This query has not been executed yet.
"""
            
            return f"{query_id} (not found in store or chat history)"
        
        return re.sub(query_pattern, replace_query, message)

    def _create_explain_prompt(self, queries: List[Dict], database_structure: Dict, focus: str) -> str:
        """Create specialized prompt for query explanation.
        
        Args:
            queries: List of expanded query details (SQL, metadata, execution plan)
            database_structure: Database schema and table information
            focus: User's specific focus/instructions
            
        Returns:
            str: Formatted prompt for LLM analysis
        """
        # Format database structure
        db_structure = []
        for table, info in database_structure.items():
            db_structure.append(f"\nTable: {table}")
            db_structure.append("Columns:")
            for col in info['columns']:
                constraints = []
                if col['pk']: constraints.append("PRIMARY KEY")
                if col['notnull']: constraints.append("NOT NULL")
                if col['default']: constraints.append(f"DEFAULT {col['default']}")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                db_structure.append(f"  - {col['name']}: {col['type']}{constraint_str}")
            
            # Add foreign key relationships
            if info.get('foreign_keys'):
                db_structure.append("\nForeign Keys:")
                for fk in info['foreign_keys']:
                    db_structure.append(f"  - {fk['from']} → {fk['table']}.{fk['to']}")
            db_structure.append("")  # Add spacing between tables
        
        # Format queries
        query_sections = []
        for i, query in enumerate(queries, 1):
            section = [f"\nQuery {i} (ID: {query['id']}):", "```sql", query['sql'], "```"]
            if 'execution_plan' in query:
                section.extend(["\nExecution Plan:", "```", query['execution_plan'], "```"])
            if 'metadata' in query:
                meta = query['metadata']
                section.append("\nMetadata:")
                section.append(f"- Tables Referenced: {', '.join(meta.get('referenced_tables', []))}")
                section.append(f"- Columns: {', '.join(meta.get('columns', []))}")
            query_sections.append("\n".join(section))

        return f"""You are a database query analyzer. Your task is to explain and analyze SQL queries, focusing on:
1. Query mechanics and structure
2. Performance implications
3. Potential improvements

Database Structure:
{chr(10).join(db_structure)}

Queries to Analyze:
{chr(10).join(query_sections)}

User's Focus: {focus}

SQL Query Guidelines:
1. Query Safety and Compatibility:
   - Use ONLY SQLite-compatible syntax
   - SQLite limitations to be aware of:
     * NO SIMILAR TO pattern matching (use LIKE or GLOB instead)
     * NO FULL OUTER JOIN (use LEFT/RIGHT JOIN)
     * NO WINDOW FUNCTIONS before SQLite 3.25
     * NO stored procedures or functions
   - NO database modifications (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE)
   - NO destructive or resource-intensive operations
   - Ensure proper table/column name quoting
   - IMPORTANT: SQL queries can ONLY be run against the connected database tables listed above

2. Query Response Format:
   IMPORTANT: You MUST follow this EXACT format for ALL SQL queries:
   a) Primary Query:
   ```sql
   -- Purpose: Clear description of query goal
   -- Tables: List of tables used
   -- Assumptions: Any important assumptions
   SELECT ... -- Your SQL here
   ```
   b) Alternative Queries (if relevant):
   ```sql
   -- Purpose: Clear description of query goal
   -- Tables: List of tables used
   -- Assumptions: Any important assumptions
   SELECT ... -- Alternative SQL
   ```
   CRITICAL FORMATTING RULES:
   1. ALWAYS include the ```sql marker at the start of EVERY SQL block
   2. ALWAYS include ALL three comment lines (Purpose, Tables, Assumptions)
   3. NEVER skip or combine the comment lines
   4. NEVER include Query IDs - they will be added automatically
   5. ALWAYS put a space after each -- in comments
   6. ALWAYS end SQL blocks with ```
   Note: Query IDs will be added automatically by the system. Do not include them in your response.

3. Query Best Practices:
   - Use explicit column names instead of SELECT *
   - Include appropriate WHERE clauses to limit results
   - Use meaningful table aliases in JOINs
   - Add comments for complex logic
   - Consider performance with large tables

IMPORTANT RULES:
- DO NOT execute SQL queries directly - only suggest them for the user to execute
- DO NOT claim to have run queries unless the user has explicitly executed them
- Ensure all SQL queries are valid for SQLite and don't use extended features

Your response should include:
1. Analysis of each query's structure and purpose
2. Performance implications and potential bottlenecks
3. Suggested improvements or alternatives (following the format above)
4. If multiple queries are provided, compare their approaches and tradeoffs

Remember to consider the user's specific focus: {focus}"""

    def _get_query_details(self, query_id: str, context: Dict[str, Any]) -> Optional[Dict]:
        """Get detailed information about a query.
        
        Args:
            query_id: ID of the query to retrieve
            context: Execution context with successful_queries_store and chat_history
            
        Returns:
            Dict with query details if found, None otherwise
        """
        # Check successful queries store first
        queries_store = context.get('successful_queries_store', {})
        stored = queries_store.get(query_id)
        
        if stored:
            return {
                'id': query_id,
                'sql': stored['sql'],
                'execution_plan': stored['metadata'].get('execution_plan', ''),
                'metadata': {
                    'referenced_tables': stored['metadata'].get('referenced_tables', []),
                    'columns': stored['metadata'].get('columns', []),
                    'rows': stored['metadata'].get('rows', 0)
                }
            }
        
        # Fall back to chat history
        query_text, found_id = self.find_recent_query(context.get('chat_history', []), query_id)
        if query_text:
            return {
                'id': query_id,
                'sql': query_text,
                'metadata': {
                    'note': 'This query has not been executed yet.'
                }
            }
        
        return None

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process natural language database queries.
        
        Required by LLMServiceMixin. Full implementation coming soon.
        For now, this is a placeholder to allow class instantiation.
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Database query processing not yet implemented")

    def _create_natural_query_prompt(self, query: str, database_structure: Dict, context: Dict) -> str:
        """Create specialized prompt for natural language to SQL translation.
        
        Args:
            query: Natural language query from user
            database_structure: Database schema and table information
            context: Execution context with chat history and query store
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Format database structure
        db_structure = []
        for table, info in database_structure.items():
            db_structure.append(f"\nTable: {table}")
            db_structure.append("Columns:")
            for col in info['columns']:
                constraints = []
                if col['pk']: constraints.append("PRIMARY KEY")
                if col['notnull']: constraints.append("NOT NULL")
                if col['default']: constraints.append(f"DEFAULT {col['default']}")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                db_structure.append(f"  - {col['name']}: {col['type']}{constraint_str}")
            
            # Add foreign key relationships
            if info.get('foreign_keys'):
                db_structure.append("\nForeign Keys:")
                for fk in info['foreign_keys']:
                    db_structure.append(f"  - {fk['from']} → {fk['table']}.{fk['to']}")
            db_structure.append("")  # Add spacing between tables
        
        # Check for query IDs in the question and get their details
        query_sections = []
        query_pattern = r'query_\d{8}_\d{6}(?:_orig|_alt\d+)'
        referenced_queries = re.findall(query_pattern, query)
        
        if referenced_queries:
            query_sections.append("\nReferenced Queries:")
            for query_id in referenced_queries:
                if details := self._get_query_details(query_id, context):
                    section = [f"\nQuery (ID: {details['id']}):", "```sql", details['sql'], "```"]
                    if 'execution_plan' in details:
                        section.extend(["\nExecution Plan:", "```", details['execution_plan'], "```"])
                    if 'metadata' in details:
                        meta = details['metadata']
                        if 'referenced_tables' in meta:
                            section.append(f"- Tables Referenced: {', '.join(meta['referenced_tables'])}")
                        if 'columns' in meta:
                            section.append(f"- Columns: {', '.join(meta['columns'])}")
                    query_sections.append("\n".join(section))

        return f"""You are a database query generator. Your task is to translate natural language questions into SQL queries.

Database Structure:
{chr(10).join(db_structure)}

{chr(10).join(query_sections) if query_sections else ""}

User's Question: {query}

SQL Query Guidelines:
1. Query Safety and Compatibility:
   - Use ONLY SQLite-compatible syntax
   - SQLite limitations to be aware of:
     * NO SIMILAR TO pattern matching (use LIKE or GLOB instead)
     * NO FULL OUTER JOIN (use LEFT/RIGHT JOIN)
     * NO WINDOW FUNCTIONS before SQLite 3.25
     * NO stored procedures or functions
   - NO database modifications (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE)
   - NO destructive or resource-intensive operations
   - Ensure proper table/column name quoting
   - IMPORTANT: SQL queries can ONLY be run against the connected database tables listed above

2. Query Response Format:
   IMPORTANT: You MUST follow this EXACT format for ALL SQL queries:
   a) Primary Query:
   ```sql
   -- Purpose: Clear description of query goal
   -- Tables: List of tables used
   -- Assumptions: Any important assumptions
   SELECT ... -- Your SQL here
   ```
   b) Alternative Queries (if relevant):
   ```sql
   -- Purpose: Clear description of query goal
   -- Tables: List of tables used
   -- Assumptions: Any important assumptions
   SELECT ... -- Alternative SQL
   ```
   CRITICAL FORMATTING RULES:
   1. ALWAYS include the ```sql marker at the start of EVERY SQL block
   2. ALWAYS include ALL three comment lines (Purpose, Tables, Assumptions)
   3. NEVER skip or combine the comment lines
   4. NEVER include Query IDs - they will be added automatically
   5. ALWAYS put a space after each -- in comments
   6. ALWAYS end SQL blocks with ```

3. Query Best Practices:
   - Use explicit column names instead of SELECT *
   - Include appropriate WHERE clauses to limit results
   - Use meaningful table aliases in JOINs
   - Add comments for complex logic
   - Consider performance with large tables

IMPORTANT RULES:
- DO NOT execute SQL queries directly - only suggest them for the user to execute
- DO NOT claim to have run queries unless the user has explicitly executed them
- Ensure all SQL queries are valid for SQLite and don't use extended features

Your response should:
1. Analyze the natural language question to understand the user's intent
2. If referenced queries are provided, consider them as context
3. Generate appropriate SQL query/queries to answer the question
4. Explain your approach and any assumptions made
5. Suggest alternative queries if there are different ways to answer the question

Remember: Users can execute your suggested queries using:
- 'search.' or 'query.' to run the primary query
- 'search query_ID' to run a specific query"""

    def get_help_text(self) -> str:
        """Get help text for database service commands."""
        return """
🔍 **Database Operations**
- First, select your database using the dropdown at the top of Data Management
- View database structure in the Database tab under Dataset Info
- View database info: `tell me about my database`
- Execute SQL:
  - Direct SQL: ```sql [your query]```
  - Last query: `search.` or `query.`
  - Specific query: `search query_[ID]` or `query query_[ID]`
- Natural language: `database: [your question]`
- Explain queries: `explain query_[ID]`
- Convert to dataset: `convert query_[ID] to dataset`
- Test service: `test database service`
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for database capabilities."""
        return """
To use the database service select and connect a database in the Data Management tab.
Database Service Commands:
1. SQL Execution:
   ```sql
   SELECT * FROM table
   ```
   - SQLite-compatible syntax only
   - No modifications (INSERT/UPDATE/DELETE/DROP)
   - Returns DataFrame results

2. Query Management:
   "search." or "query." - run last query
   "search query_[ID]" or "query query_[ID]" - run specific query
   "explain query_[ID]" - analyze query
   "database: [question]" - natural language to SQL
   "test database service" - run service diagnostics

3. Result Conversion:
   "convert query_[ID] to dataset"
   - Saves query results as new dataset
   - Preserves query and execution info"""
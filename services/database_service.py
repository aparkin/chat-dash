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

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    ServiceContext,
    PreviewIdentifier
)

class DatabaseService(ChatService):
    """Service for database query handling and execution."""
    
    def __init__(self):
        super().__init__("database")
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
        """
        # Clean and normalize message
        message = message.strip()
        
        # Check for SQL code blocks
        if self.sql_block_re.search(message):
            return True
            
        # Check for execution commands and test command
        for pattern in self.execution_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
                
        return False
    
    def parse_request(self, message: str) -> dict:
        """Parse database service request from message.
        
        Handles:
        - Query search commands (search., query.)
        - Specific query search (search query_ID)
        - Query conversion (convert query_ID to dataset)
        - Database info requests (tell me about my database)
        
        Returns dict with parsed parameters.
        """
        # Normalize message
        message = message.lower().strip()
        
        # Check for simple execution command
        if message in ['search.', 'query.']:
            return {
                'command': 'execute',
                'query_id': None  # Will use most recent query
            }
            
        # Check for specific query execution
        execute_match = re.match(r'^(?:search|query)\s+(query_\d{8}_\d{6}(?:_orig|_alt\d+)?)\b', message)
        if execute_match:
            return {
                'command': 'execute',
                'query_id': execute_match.group(1)
            }
            
        # Check for query conversion
        convert_pattern = r'^convert\s+(query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b'
        convert_match = re.match(convert_pattern, message)
        if convert_match:
            return {
                'command': 'convert',
                'query_id': convert_match.group(1)
            }
            
        # Check for database info request
        if message == 'tell me about my database':
            return {
                'command': 'info'
            }
            
        return {}

    def find_recent_query(self, chat_history: list, query_id: str = None) -> tuple[str, str]:
        """Find SQL query in chat history."""
        print(f"\nSearching for query in chat history...")
        print(f"Target query ID: {query_id}")
        
        for msg in reversed(chat_history):
            if msg['role'] == 'assistant' and '```sql' in msg['content'].lower():
                content = msg['content']
                print(f"\nFound SQL block in message:")
                print(f"Message role: {msg['role']}")
                
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
    
    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute database service request.
        
        Args:
            params: Parameters from parse_request
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
            # Check database connection
            if not context.get('database_state', {}).get('connected'):
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No database connected. Please connect to a database first.",
                        message_type="error"
                    )]
                )
            
            db_path = context['database_state']['path']
            command = params.get('command')
            
            if command == 'execute':
                # Find query to execute
                query_text, query_id = self.find_recent_query(
                    context['chat_history'],
                    params.get('query_id')
                )
                
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No SQL query found in recent chat history. Please make sure a query has been suggested before using 'search.' or 'query.'",
                            message_type="error"
                        )]
                    )
                
                # Execute query
                try:
                    results, metadata, preview = self._execute_sql_query(query_text, db_path)
                    
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
                    
                    # Create focused context for result analysis
                    service_context = ServiceContext(
                        source=self.name,
                        data={
                            'service_type': 'database',
                            'command_state': {
                                'type': 'query_execution',
                                'status': 'completed',
                                'query_id': query_id,
                                'query_text': query_text,
                                'execution_time': datetime.now().isoformat()
                            },
                            'results': {
                                'preview': results.head().to_dict('records'),
                                'total_rows': len(results),
                                'columns': list(results.columns),
                                'referenced_tables': metadata.get('referenced_tables', []),
                                'execution_plan': metadata['execution_plan']
                            },
                            'action': 'query_execution',
                            'status': 'completed'
                        },
                        metadata={
                            'task': 'analyze_query_results',
                            'execution_status': 'completed',
                            'service_action': 'database_query',
                            'analysis_prompts': [
                                "This is a completed database query. Please analyze the SQL query results:",
                                "- Summarize the key information shown in the results",
                                "- Note any patterns or interesting findings in the data",
                                "- Suggest relevant follow-up queries or analyses based on these results",
                                "- Identify relationships with other tables or data sources",
                                "- Consider the execution plan for performance insights",
                                f"Note: You can convert these results to a dataset using 'convert {query_id} to dataset'"
                            ]
                        }
                    )
                    
                    print("\n=== Execute Command Response ===")
                    print(f"Has context: {bool(service_context)}")
                    print(f"Context source: {service_context.source}")
                    print("Context data:")
                    for key, value in service_context.data.items():
                        print(f"  {key}: {value}")
                    print("Context metadata:")
                    for key, value in service_context.metadata.items():
                        print(f"  {key}: {value}")
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type="info"
                        )],
                        context=service_context,
                        store_updates=store_updates,
                        state_updates={'chat_input': ''}
                    )
                    
                except sqlite3.Error as e:
                    # Handle SQL errors
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"SQL Error: {str(e)}",
                            message_type="error"
                        )],
                        state_updates={'chat_input': ''}
                    )
                except Exception as e:
                    # Handle other errors
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error: {str(e)}",
                            message_type="error"
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif command == 'convert':
                # Handle dataset conversion in a separate method
                return self._handle_dataset_conversion(params, context)
                
            elif command == 'info':
                # Get database structure from context
                structure = context.get('database_structure', {})
                if not structure:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="Database structure information not available.",
                            message_type="error"
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
                
                print("\n=== Info Command Response ===")
                print(f"Structure keys: {list(structure.keys())}")
                print("Sample table info:")
                if structure:
                    sample_table = next(iter(structure.items()))
                    print(f"  Table: {sample_table[0]}")
                    print(f"  Row count: {sample_table[1]['row_count']}")
                    print(f"  Columns: {len(sample_table[1]['columns'])}")
                print("Message content preview:")
                print("\n".join(overview[:5]) + "...")

                # Create focused context for database structure
                service_context = ServiceContext(
                    source=self.name,
                    data={
                        'service_type': 'database',
                        'command_state': {
                            'type': 'info',
                            'status': 'completed',
                            'execution_time': datetime.now().isoformat()
                        },
                        'structure': structure,  # Pass the entire structure directly
                        'action': 'database_info',  # Add required action field
                        'status': 'completed'      # Add required status field
                    },
                    metadata={
                        'task': 'database_overview',
                        'execution_status': 'completed',
                        'service_action': 'database_info',
                        'analysis_prompts': [
                            "This is a database structure overview. Please analyze:",
                            "- Summarize the key tables and their purposes based on their structure",
                            "- Identify and explain important relationships between tables",
                            "- Note any interesting schema design patterns or constraints",
                            "- Suggest relevant queries that could be run to explore the data",
                            "- Identify potential data analysis opportunities based on the schema",
                            "- Consider performance implications of the current structure"
                        ]
                    }
                )
                
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="\n".join(overview),
                        message_type="info"
                    )],
                    context=service_context,
                    state_updates={'chat_input': ''}
                )
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Invalid command",
                    message_type="error"
                )],
                state_updates={'chat_input': ''}  # Clear input even on error
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Service error: {str(e)}",
                    message_type="error"
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
            result = self.parse_request(message)
            assert result['type'] == 'direct_sql'
            assert result['sql'] == 'SELECT * FROM table'
        
        run_test("SQL parsing", test_sql_parsing)
        
        # 2. Test command parsing
        def test_command_parsing():
            message = "execute query_20240101_123456_orig"
            result = self.parse_request(message)
            assert result['type'] == 'execute_query'
            assert result['query_id'] == 'query_20240101_123456_orig'
        
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
                {'type': 'direct_sql', 'sql': 'INVALID SQL'},
                {'database_state': {'connected': True, 'path': ':memory:'}}
            )
            assert response.messages[0].message_type == "error"
            assert "failed" in response.messages[0].content.lower()
        
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
                message_type="info" if passed == total else "warning"
            )],
            context=ServiceContext(
                source=self.name,
                data={
                    'test_results': {
                        'passed': passed,
                        'total': total,
                        'details': test_results
                    }
                },
                metadata={'test_timestamp': datetime.now().isoformat()}
            )
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
        
        Adds a Query ID comment at the start of each SQL block:
        ```sql
        -- Query ID: query_YYYYMMDD_HHMMSS_[orig|altN]
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
                
                # Add ID comment at the start of the block
                block_parts = text[start:end].split(sql_content)
                modified_block = f"{block_parts[0]}-- Query ID: {query_id}\n{sql_content}{block_parts[1]}"
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
                    message_type="error"
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
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )
        
        try:
            # Re-execute query to get fresh results
            db_path = context['database_state']['path']
            results, metadata, _ = self._execute_sql_query(stored['sql'], db_path)
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
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
                    'referenced_tables': metadata.get('referenced_tables', [])
                }
            }

            # Create focused context for conversion
            service_context = ServiceContext(
                source=self.name,
                data={
                    'service_type': 'database',
                    'command_state': {
                        'type': 'dataset_conversion',
                        'status': 'completed',
                        'query_id': query_id,
                        'execution_time': datetime.now().isoformat()
                    },
                    'conversion_result': {
                        'dataset_name': query_id,
                        'rows': len(results),
                        'columns': list(results.columns),
                        'source': {
                            'type': 'sql_query',
                            'query_id': query_id,
                            'query_text': stored['sql'],
                            'execution_plan': metadata['execution_plan']
                        },
                        'referenced_tables': metadata.get('referenced_tables', []),
                        'preview': results.head().to_dict('records')
                    },
                    'action': 'dataset_conversion',
                    'status': 'completed'
                },
                metadata={
                    'task': 'query_result_conversion',
                    'execution_status': 'completed',
                    'service_action': 'dataset_conversion',
                    'analysis_prompts': [
                        "This query result has been converted to a dataset. Please analyze:",
                        "- Summarize the key characteristics of the converted dataset",
                        "- Explain how this dataset relates to its source tables",
                        "- Note any important patterns or distributions in the preview data",
                        "- Suggest potential analyses that could be performed with this dataset",
                        "- Identify opportunities to combine this with other available data sources",
                        "- Consider any data quality or completeness implications"
                    ]
                }
            )
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Query results converted to dataset '{query_id}'

- Rows: {len(results)}
- Columns: {', '.join(results.columns)}
- Source: SQL Query {query_id}
- Referenced tables: {', '.join(metadata.get('referenced_tables', []))}""",
                    message_type="info"
                )],
                context=service_context,
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
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )
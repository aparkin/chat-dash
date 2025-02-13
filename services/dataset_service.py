"""
Dataset service implementation.

This service handles dataset analysis and code execution in the ChatDash application.
It provides a modular interface for:
1. Dataset information and exploration
2. Safe code execution for analysis
3. Dataset creation from analysis results
4. Dataset state management
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import re
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from pathlib import Path
from ydata_profiling import ProfileReport
import traceback

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    ServiceContext,
    PreviewIdentifier
)

class DatasetAnalysisException(Exception):
    """Base exception for dataset analysis errors."""
    pass

class ValidationError(DatasetAnalysisException):
    """Raised when code validation fails."""
    pass

class ExecutionError(DatasetAnalysisException):
    """Raised when code execution fails."""
    pass

class SecurityError(DatasetAnalysisException):
    """Raised when code violates security constraints."""
    pass

class CodeValidator:
    """Validates code blocks for safety and correctness."""
    
    def __init__(self):
        # Allowed module imports
        self.allowed_imports = {
            'pandas', 'numpy', 'plotly', 
            'scipy.stats', 'sklearn.preprocessing'
        }
        
        # Blocked operations/attributes
        self.blocked_ops = {
            'eval', 'exec', 'compile',
            'open', 'file', 'os', 'sys',
            'subprocess', 'import', '__import__'
        }
        
        # Compile regex patterns
        self.import_pattern = re.compile(r'^(?:from|import)\s+([a-zA-Z0-9_.]+)')
        self.blocked_pattern = re.compile(
            r'(?:' + '|'.join(map(re.escape, self.blocked_ops)) + r')\s*\('
        )
    
    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code block for safety and correctness.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Check for blocked operations
            if self.blocked_pattern.search(code):
                return False, "Code contains blocked operations"
            
            # Check imports
            for line in code.split('\n'):
                if match := self.import_pattern.match(line.strip()):
                    module = match.group(1).split('.')[0]
                    if module not in self.allowed_imports:
                        return False, f"Import of '{module}' not allowed"
            
            # Try to compile code to check syntax
            compile(code, '<string>', 'exec')
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class CodeExecutor:
    """Executes validated code blocks safely."""
    
    def __init__(self):
        self.validator = CodeValidator()
        
        # Initialize execution environment with all necessary libraries
        self.globals = {
            # Core data libraries
            'pd': pd,
            'np': np,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            
            # Plotting libraries
            'px': px,
            'go': go,
            'make_subplots': make_subplots,
            
            # Statistics and preprocessing
            'stats': stats,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            
            # Constants and utilities
            'np.nan': np.nan,
            'np.inf': np.inf,
            'datetime': datetime
        }
        
        # Add common pandas functions
        for func in ['concat', 'merge', 'qcut', 'cut', 'to_datetime', 'to_numeric']:
            self.globals[func] = getattr(pd, func)
        
        # Add common numpy functions
        for func in ['mean', 'median', 'std', 'min', 'max', 'sum', 'abs', 'log', 'exp']:
            self.globals[func] = getattr(np, func)
    
    def execute(self, code: str, df: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
        """Execute code safely in a controlled environment.
        
        Args:
            code: Python code to execute
            df: DataFrame to make available to the code
            
        Returns:
            Tuple[Any, Dict[str, Any]]: (execution_result, metadata)
        """
        # First validate the code
        is_valid, error = self.validator.validate(code)
        if not error:
            # Create a local namespace for execution
            local_vars = {
                'df': df,
                'result': None,
                'viz_data': None
            }
            
            try:
                # Execute the code
                exec(code, self.globals, local_vars)
                
                # Get the result (either explicitly set or last expression)
                result = local_vars.get('result', None)
                
                # Collect metadata about the execution
                metadata = {
                    'execution_time': datetime.now().isoformat(),
                    'code_size': len(code),
                    'has_viz': 'viz_data' in local_vars and local_vars['viz_data'] is not None
                }
                
                # If there's visualization data, include it in metadata
                if metadata['has_viz']:
                    metadata['viz_data'] = local_vars['viz_data']
                
                return result, metadata
                
            except Exception as e:
                raise ExecutionError(f"Code execution failed: {str(e)}")
        else:
            raise ValidationError(f"Code validation failed: {error}")

class DatasetService(ChatService):
    """Service for dataset analysis and code execution."""
    
    def __init__(self):
        super().__init__("dataset")
        # Register our prefixes
        PreviewIdentifier.register_prefix("datasetCode")  # For code blocks
        PreviewIdentifier.register_prefix("dataset")      # For datasets
        
        # Code block pattern
        self.code_block_pattern = r'```\s*python\s*\n(.*?)```'  # More lenient pattern
        
        # Command patterns
        self.command_patterns = [
            # Dataset info requests
            r'tell\s+me\s+about\s+my\s+datasets?\b',  # Handle both singular and plural
            r'analyze\s+dataset\s+(\w+)\b',
            r'^analysis:\s*(.+)$',  # New pattern for analysis requests
            
            # Code execution
            r'^(?:run|execute)\s+((?i:datasetCode)_\d{8}_\d{6}(?:_orig|_alt\d+))\b',
            r'^(?:run|execute)[.!]$',
            
            # Dataset conversion
            r'convert\s+dataset_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b'
        ]
        
        # Compile patterns for efficiency
        self.code_block_re = re.compile(self.code_block_pattern, re.IGNORECASE | re.DOTALL)
        self.command_res = [re.compile(p, re.IGNORECASE) for p in self.command_patterns]
        
        # Initialize code execution components
        self.executor = CodeExecutor()
    
    def can_handle(self, message: str) -> bool:
        """Detect if message contains dataset commands.
        
        Handles:
        1. Python code blocks (```python df.head()```)
        2. Dataset info requests (tell me about my datasets)
        3. Code execution commands (run datasetCode_ID)
        4. Dataset conversion (convert dataset_ID to dataset)
        """
        # Clean and normalize message
        message = message.strip()
        
        # Check for Python code blocks
        if self.code_block_re.search(message):
            return True
            
        # Check for dataset commands
        for pattern in self.command_res:
            if pattern.search(message):
                return True
                
        return False
    
    def parse_request(self, message: str) -> dict:
        """Parse dataset service request from message."""
        # Normalize message
        message = message.lower().strip()
        print(f"\nParsing dataset request: '{message}'")
        
        # Check for simple execution command first
        if message in ['run.', 'execute.', 'run!', 'execute!']:
            print("Detected execution command")
            return {
                'command': 'execute',
                'code_id': None  # Will use most recent code block
            }
        
        # Check for specific code execution
        execute_match = re.match(
            r'^(?:run|execute)\s+((?i:datasetCode)_\d{8}_\d{6}(?:_orig|_alt\d+))\b',
            message
        )
        print(f"execute_match: {execute_match}, message: {message}")
        if execute_match:
            print(f"Detected specific code execution: {execute_match.group(1)}")
            return {
                'command': 'execute',
                'code_id': execute_match.group(1)
            }
            
        # Check for dataset conversion
        convert_match = re.match(
            r'^convert\s+(dataset_\d{8}_\d{6}(?:_orig|_alt\d+)?)\s+to\s+dataset\b',
            message
        )
        if convert_match:
            return {
                'command': 'convert',
                'code_id': convert_match.group(1)
            }
            
        # Check for dataset info request
        info_match = re.match(r'tell\s+me\s+about\s+my\s+datasets?\b', message)
        if info_match:
            is_plural = 'datasets' in info_match.group(0)
            return {
                'command': 'info',
                'target': 'all' if is_plural else 'selected'
            }
            
        # Check for specific dataset analysis
        analyze_match = re.match(r'analyze\s+dataset\s+(\w+)\b', message)
        if analyze_match:
            return {
                'command': 'info',
                'target': analyze_match.group(1)
            }
            
        # Check for analysis request
        analysis_match = re.match(r'^analysis:\s*(.+)$', message)
        if analysis_match:
            return {
                'command': 'analysis',
                'description': analysis_match.group(1).strip()
            }
            
        # Check for Python code block
        if self.code_block_re.search(message):
            return {
                'command': 'validate',
                'code': message
            }
            
        return {}
    
    def detect_content_blocks(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect Python code blocks in text.
        
        Returns:
            List of tuples (code_content, start_pos, end_pos)
        """
        blocks = []
        for match in self.code_block_re.finditer(text):
            start, end = match.span()
            content = match.group(1).strip()
            blocks.append((content, start, end))
        return blocks
        
    def add_ids_to_blocks(self, text: str) -> str:
        """Add code IDs to Python blocks in text.
        
        Adds a Code ID comment at the start of each Python block:
        ```python
        # Code ID: datasetCode_YYYYMMDD_HHMMSS_[orig|altN]
        df.head()
        ```
        """
        if '```python' not in text.lower():
            return text
            
        modified_response = []
        current_pos = 0
        
        # Get base ID for this response
        base_id = PreviewIdentifier.create_id(prefix="datasetCode")
        previous_id = base_id  # Track the last generated ID
        block_counter = 0
        
        # Process each Python block
        for code_content, start, end in self.detect_content_blocks(text):
            # Add text before this block
            modified_response.append(text[current_pos:start])
            
            # Skip if block already has a Code ID
            if re.search(r'^#\s*Code ID:', code_content):
                modified_response.append(text[start:end])
            else:
                # Generate new ID based on counter
                if block_counter == 0:
                    code_id = base_id
                else:
                    # Use the previous ID to generate the next alternative
                    code_id = PreviewIdentifier.create_id(previous_id=previous_id)
                
                previous_id = code_id  # Update previous_id for next iteration
                block_counter += 1
                
                # Add ID comment at the start of the block
                block_parts = text[start:end].split(code_content)
                modified_block = f"{block_parts[0]}# Code ID: {code_id}\n{code_content}{block_parts[1]}"
                modified_response.append(modified_block)
            
            current_pos = end
        
        # Add any remaining text
        modified_response.append(text[current_pos:])
        
        return ''.join(modified_response)
    
    def _run_self_test(self, context: Dict[str, Any]) -> ServiceResponse:
        """Run self-tests for dataset service functionality.
        
        This method tests core functionality within the current application context:
        1. Code validation and safety checks
        2. Code execution (if dataset available)
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
        
        # 1. Test code validation
        def test_code_validation():
            # Test valid code
            code = "result = df.head()"
            is_valid, error = self.executor.validator.validate(code)
            assert is_valid
            assert error is None
            
            # Test blocked operations
            code = "result = eval('1+1')"
            is_valid, error = self.executor.validator.validate(code)
            assert not is_valid
            assert "blocked operations" in error.lower()
        
        run_test("Code validation", test_code_validation)
        
        # 2. Test command parsing
        def test_command_parsing():
            # Test info request
            result = self.parse_request("tell me about my datasets")
            assert result['command'] == 'info'
            assert result['target'] == 'all'
            
            # Test code execution
            result = self.parse_request("run datasetCode_20240101_123456_orig")
            assert result['command'] == 'execute'
            assert result['code_id'] == 'datasetCode_20240101_123456_orig'
        
        run_test("Command parsing", test_command_parsing)
        
        # 3. Test code execution
        if context.get('datasets_store') and context.get('selected_dataset'):
            def test_code_execution():
                df = pd.DataFrame({'a': [1, 2, 3]})
                result, metadata = self.executor.execute("result = df.describe()", df)
                assert isinstance(result, pd.DataFrame)
                assert 'mean' in result.index
            
            run_test("Code execution", test_code_execution)
        else:
            test_results.append("⚠ Skipped code execution tests - no dataset available")
        
        # 4. Test error handling
        def test_error_handling():
            # Test invalid code execution
            response = self.execute(
                {'command': 'execute', 'code': 'invalid code'},
                {'datasets_store': {}, 'selected_dataset': None}
            )
            assert response.messages[0].message_type == "error"
            assert "No datasets" in response.messages[0].content
        
        run_test("Error handling", test_error_handling)
        
        # Format results
        summary = [
            "### Dataset Service Self-Test Results",
            f"\nPassed: {passed}/{total} tests\n",
            "Detailed Results:"
        ] + test_results
        
        if context.get('datasets_store') and context.get('selected_dataset'):
            summary.append("\nTests run with active dataset")
        else:
            summary.append("\n⚠ Some tests skipped - no dataset available")
        
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

    def _handle_info_request(self, params: dict, context: dict) -> ServiceResponse:
        """Handle dataset information requests."""
        datasets = context.get('datasets_store', {})
        if not datasets:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No datasets are currently loaded.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        target = params.get('target', 'all')
        
        # Handle 'selected' target type
        if target == 'selected':
            selected_dataset = context.get('selected_dataset')
            if not selected_dataset:
                target = 'all'
            else:
                target = selected_dataset
        
        if target == 'all':
            # Generate overview of all datasets
            parts = ["### Dataset Overview"]
            
            # Add dataset summaries
            for name, data in datasets.items():
                df = pd.DataFrame(data['df'])
                metadata = data['metadata']
                parts.extend([
                    f"\n**{name}**",
                    f"- Source: {metadata['source']}",
                    f"- Upload time: {metadata['upload_time']}",
                    f"- Rows: {len(df)}",
                    f"- Columns: {len(df.columns)}"
                ])
            
            # Create focused context
            service_context = ServiceContext(
                source=self.name,
                data={
                    'datasets': {
                        name: {
                            'rows': len(pd.DataFrame(data['df'])),
                            'columns': len(pd.DataFrame(data['df']).columns)
                        }
                        for name, data in datasets.items()
                    }
                },
                metadata={
                    'task': 'dataset_overview',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="\n".join(parts),
                    message_type="info",
                    role="assistant"
                )],
                context=service_context,
                state_updates={'chat_input': ''}
            )
            
        else:
            # Show detailed info for specific dataset
            if target not in datasets:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Dataset '{target}' not found.",
                        message_type="error",
                        role="assistant"
                    )],
                    state_updates={'chat_input': ''}
                )
            
            df = pd.DataFrame(datasets[target]['df'])
            metadata = datasets[target]['metadata']
            
            # Build analysis parts
            parts = [
                f"### Analysis of Dataset: {target}",
                "\n**Basic Information**",
                f"- Source: {metadata['source']}",
                f"- Upload time: {metadata['upload_time']}",
                f"- Rows: {len(df)}",
                f"- Columns: {len(df.columns)}",
                "\n**Column Information**"
            ]
            
            # Add column details
            for col in df.columns:
                dtype = df[col].dtype
                n_unique = df[col].nunique()
                n_missing = df[col].isna().sum()
                
                parts.extend([
                    f"\n{col}:",
                    f"- Type: {dtype}",
                    f"- Unique values: {n_unique}",
                    f"- Missing values: {n_missing}"
                ])
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = df[col].describe()
                    parts.extend([
                        f"- Range: {stats['min']:.2f} to {stats['max']:.2f}",
                        f"- Mean: {stats['mean']:.2f}",
                        f"- Std: {stats['std']:.2f}"
                    ])
            
            # Create focused context
            service_context = ServiceContext(
                source=self.name,
                data={
                    'dataset_name': target,
                    'analysis_summary': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
                        'categorical_cols': df.select_dtypes(exclude=[np.number]).columns.tolist()
                    }
                },
                metadata={
                    'analysis_time': datetime.now().isoformat(),
                    'analysis_type': 'dataset_overview'
                }
            )
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="\n".join(parts),
                    message_type="info",
                    role="assistant"
                )],
                context=service_context,
                state_updates={'chat_input': ''}
            )
    
    def _handle_code_validation(self, params: dict, context: dict) -> ServiceResponse:
        """Handle code block validation."""
        # Extract code blocks
        blocks = self.detect_content_blocks(params['code'])
        if not blocks:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No Python code blocks found in message.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        # Validate each block
        responses = []
        for code, _, _ in blocks:
            is_valid, error = self.executor.validator.validate(code)
            if not is_valid:
                responses.append(ServiceMessage(
                    service=self.name,
                    content=f"❌ Code validation failed: {error}",
                    message_type="error",
                    role="assistant"
                ))
            else:
                responses.append(ServiceMessage(
                    service=self.name,
                    content="✓ Code validation successful",
                    message_type="info",
                    role="assistant"
                ))
        
        return ServiceResponse(
            messages=responses,
            state_updates={'chat_input': ''}
        )
    
    def _handle_code_execution(self, params: dict, context: dict) -> ServiceResponse:
        """Handle code execution requests."""
        # Validate environment
        validation_result = self._validate_execution_environment(params, context)
        if not isinstance(validation_result, tuple):
            return validation_result
        df, selected_dataset, code, code_id = validation_result

        try:
            # Execute code in isolated environment
            result, metadata = self.executor.execute(code, df)
            
            # Process results and build response components
            store_updates = {}
            state_updates = {'chat_input': ''}
            
            # Handle visualization if present
            if metadata.get('has_viz') and metadata.get('viz_data'):
                viz_data = metadata['viz_data']
                if isinstance(viz_data, dict) and 'figure' in viz_data:
                    state_updates['active_tab'] = 'tab-viz'
                    state_updates['viz_state'] = self._process_visualization(viz_data, selected_dataset, df, context)

            # Process results based on type
            if isinstance(result, dict) and all(isinstance(v, (pd.DataFrame, type(None))) for v in result.values()):
                # Handle multiple DataFrame results
                result_ids, preview_text, context_data = self._process_multiple_results(result, code_id, selected_dataset, store_updates)
                main_result_id = result_ids[0] if result_ids else None
                
                # Create service context
                service_context = ServiceContext(
                    source=self.name,
                    data=context_data,
                    metadata={
                        'execution_time': datetime.now().isoformat(),
                        'result_count': len(result),
                        'analysis_prompts': [
                            "Please analyze the results of the code execution:",
                            "- Summarize what each DataFrame contains",
                            "- Note any interesting patterns or relationships between the results",
                            "- Suggest potential next steps for analysis"
                        ]
                    }
                )
            else:
                # Handle single result (existing logic)
                result_id, preview_text = self._process_result(result, code_id, selected_dataset, store_updates)
                result_ids = [result_id]
                main_result_id = result_id
                
                if isinstance(result, pd.DataFrame):
                    result_data = result.to_dict('records')
                elif isinstance(result, pd.Series):
                    result_data = result.to_frame().to_dict('records')
                else:
                    result_data = {'value': str(result)}
                
                # Create service context
                service_context = ServiceContext(
                    source=self.name,
                    data=result_data,
                    metadata={
                        'execution_time': datetime.now().isoformat(),
                        'result_count': 1,
                        'analysis_prompts': [
                            "Please analyze the result of the code execution:",
                            "- Summarize the result",
                            "- Note any interesting patterns or insights"
                        ]
                    }
                )
            
            # Build response
            response = self._build_execution_response(
                preview_text,
                main_result_id,
                bool(state_updates.get('viz_state'))
            )

            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=response,
                    message_type="info",
                    role="assistant"
                )],
                context=service_context,
                store_updates=store_updates,
                state_updates=state_updates
            )

        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Dataset service error in code execution: {str(e)}",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )

    def _validate_execution_environment(self, params: dict, context: dict) -> Union[ServiceResponse, Tuple[pd.DataFrame, str, str, str]]:
        """Validate execution environment and extract required components."""
        # Check for selected dataset
        selected_dataset = context.get('selected_dataset')
        datasets = context.get('datasets_store', {})
        
        if not datasets:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Dataset service error in validation: No datasets are currently loaded.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        if not selected_dataset:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset is selected.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get code to execute
        code_id = params.get('code_id')
        code = self._find_code_block(context['chat_history'], code_id)
        
        if not code:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No Python code block found in recent chat history.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get dataset
        df = pd.DataFrame(datasets[selected_dataset]['df'])
        df.name = selected_dataset
        
        return df, selected_dataset, code, code_id

    def _process_visualization(self, viz_data: dict, dataset_name: str, df: pd.DataFrame, context: dict) -> Optional[dict]:
        """Process visualization data and create visualization state."""
        if not viz_data or 'figure' not in viz_data:
            return None
            
        # Ensure figure is converted to dictionary if it's a plotly figure
        figure = viz_data['figure']
        if hasattr(figure, 'to_dict'):
            figure = figure.to_dict()
            
        # Return state for direct visualization in the viz-container
        return {
            'figure': figure,  # This will be used directly by viz-container
            'type': 'plotly',
            'params': {
                'dataset_name': dataset_name,
                'rows': len(df),
                'columns': len(df.columns)
            },
            'view_settings': context.get('viz_state', {}).get('view_settings', {})
        }

    def _process_result(self, result: Any, code_id: str, dataset_name: str, store_updates: dict) -> Tuple[str, str]:
        """Process execution result and prepare preview."""
        if isinstance(result, pd.DataFrame):
            result_id = PreviewIdentifier.create_id(prefix="dataset")
            preview = f"DataFrame Result ({len(result)} rows × {len(result.columns)} columns):\n"
            preview += f"Dataset ID: {result_id}\n"
            preview += f"```\n{result.head().to_string()}\n```"
            
            store_updates['successful_queries_store'] = {
                result_id: {
                    'type': 'dataframe',
                    'code_id': code_id,
                    'dataframe': result.to_dict('records'),
                    'metadata': {
                        'source_dataset': dataset_name,
                        'execution_time': datetime.now().isoformat(),
                        'rows': len(result),
                        'columns': list(result.columns)
                    }
                }
            }
        elif isinstance(result, pd.Series):
            result_id = PreviewIdentifier.create_id(prefix="dataset")
            preview = f"Series Result ({len(result)} elements):\n"
            preview += f"Dataset ID: {result_id}\n"
            preview += f"```\n{result.head().to_string()}\n```"
            
            store_updates['successful_queries_store'] = {
                result_id: {
                    'type': 'series',
                    'code_id': code_id,
                    'dataframe': result.to_frame().to_dict('records'),
                    'metadata': {
                        'source_dataset': dataset_name,
                        'execution_time': datetime.now().isoformat(),
                        'rows': len(result),
                        'name': result.name or 'value'
                    }
                }
            }
        else:
            result_id = PreviewIdentifier.create_id(prefix="dataset")
            preview = f"Result:\n```\n{str(result)[:1000]}\n```"
            
            store_updates['successful_queries_store'] = {
                result_id: {
                    'type': 'other',
                    'code_id': code_id,
                    'result': str(result),
                    'metadata': {
                        'source_dataset': dataset_name,
                        'execution_time': datetime.now().isoformat(),
                        'result_type': type(result).__name__
                    }
                }
            }
            
        return result_id, preview

    def _build_execution_response(self, result_preview: str, result_id: str, has_viz: bool) -> str:
        """Build consistent execution response message."""
        parts = [
            "Code executed successfully!",
            result_preview
        ]
        
        if has_viz:
            parts.append("\nVisualization created and available in visualization tab.")
            
        parts.extend([
            "\nAvailable Actions:",
            f"1. Save result as dataset: convert {result_id} to dataset",
            "2. Run additional analysis",
            "3. View in visualization tab" if has_viz else "3. Create visualization",
            "\nUse these actions by typing the corresponding command in the chat."  # Clear closing line
        ])
        
        return "\n".join(parts)

    def _find_code_block(self, chat_history: List[dict], code_id: Optional[str] = None) -> Optional[str]:
        """Find code block in chat history.
        
        Args:
            chat_history: List of chat messages
            code_id: Optional specific code ID to find (e.g. datasetCode_20250212_073207_orig)
            
        Returns:
            The code block if found, None otherwise
        """
        print("\n=== Debug: Finding Code Block ===")
        print(f"Requested code_id: {code_id}")
        print(f"Chat history length: {len(chat_history)}")
        
        for msg in reversed(chat_history):
            if msg['role'] == 'assistant' and '```python' in msg['content'].lower():
                print("\nFound message with Python block")
                for block, _, _ in self.detect_content_blocks(msg['content']):
                    lines = block.split('\n')
                    if not lines:
                        continue
                        
                    # Look for the Code ID in the first line
                    first_line = lines[0].strip()
                    print(f"Checking first line: {first_line}")
                    
                    # Check if line starts with # Code ID: and extract the ID
                    if first_line.startswith('# Code ID:'):
                        found_id = first_line.replace('# Code ID:', '').strip()
                        print(f"Found code block with ID: {found_id}")
                        if code_id:
                            if found_id.lower() == code_id.lower():  # Case-insensitive comparison
                                print(f"Matched requested ID: {code_id}")
                                return block
                        else:
                            # For run., return the first (most recent) block found
                            print("No specific ID requested, returning this block")
                            return block
                            
        print("No matching code block found")
        return None

    def _handle_dataset_conversion(self, params: dict, context: dict) -> ServiceResponse:
        """Handle dataset conversion requests."""
        print("\n=== Dataset Conversion Debug ===")
        dataset_id = params.get('code_id')  # This is actually the dataset ID
        if not dataset_id:
            print("Error: No dataset ID provided")
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset ID provided for conversion.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        print(f"Converting dataset ID: {dataset_id}")
        
        # Get stored execution
        executions = context.get('successful_queries_store', {})
        stored = executions.get(dataset_id)
        
        if not stored:
            print(f"Error: No execution found for dataset ID: {dataset_id}")
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"No execution found for dataset ID: {dataset_id}",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        print("Found stored execution data")
        print(f"Stored data keys: {list(stored.keys())}")
        
        try:
            # Convert result to DataFrame
            if 'dataframe' in stored:
                print("Converting from stored dataframe")
                df = pd.DataFrame(stored['dataframe'])
            else:
                print("Converting from stored result")
                result = stored['result']
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                elif isinstance(result, dict):
                    df = pd.DataFrame.from_dict(result)
                else:
                    df = pd.DataFrame(result)
            
            print(f"Successfully created DataFrame: {len(df)} rows × {len(df.columns)} columns")
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
            # Generate profile report
            try:
                profile = ProfileReport(
                    df,
                    minimal=True,
                    title=f"Profile Report for {dataset_id}",
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
            datasets[dataset_id] = {
                'df': df.to_dict('records'),  # Preserve original structure
                'metadata': {
                    'source': f"Code execution: {dataset_id}",
                    'original_dataset': stored['metadata']['source_dataset'],
                    'code': stored.get('code', 'Generated result'),
                    'result_type': stored['metadata'].get('result_type', 'dataframe'),
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df),
                    'columns': list(df.columns)
                },
                'profile_report': profile_html
            }
            
            print(f"Dataset created with ID: {dataset_id}")
            print("Dataset structure:")
            print(f"- Has df: {bool(datasets[dataset_id]['df'])}")
            print(f"- Has metadata: {bool(datasets[dataset_id]['metadata'])}")
            print(f"- Has profile: {bool(datasets[dataset_id]['profile_report'])}")
            
            response = ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Code results converted to dataset '{dataset_id}'

- Rows: {len(df)}
- Columns: {', '.join(df.columns)}
- Source: Code {dataset_id}
- Type: {stored['metadata'].get('result_type', 'dataframe')}""",
                    message_type="info",
                    role="assistant"
                )],
                store_updates={
                    'datasets_store': datasets,  # Update datasets store
                    'successful_queries_store': {  # Update queries store, removing the converted query
                        k: v for k, v in executions.items() if k != dataset_id
                    }
                },
                state_updates={'chat_input': ''}
            )
            
            print("\nCreated response:")
            print(f"- Has messages: {bool(response.messages)}")
            print(f"- Store updates: {list(response.store_updates.keys())}")
            print(f"- State updates: {list(response.state_updates.keys())}")
            
            return response
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"{self.name} service error: ❌ Dataset conversion failed: {str(e)}",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )

    def _handle_analysis_request(self, params: dict, context: dict) -> ServiceResponse:
        """Handle analysis requests with code generation."""
        # Validate dataset selection
        selected_dataset = context.get('selected_dataset')
        datasets = context.get('datasets_store', {})
        
        if not datasets or not selected_dataset:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset is selected.",
                    message_type="error",
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
            
        # Get dataset info
        df = pd.DataFrame(datasets[selected_dataset]['df'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Generate code ID
        code_id = PreviewIdentifier.create_id(prefix="datasetCode")
        
        # Create analysis code with embedded ID
        analysis_code = f"""# Code ID: {code_id}
# Analysis for dataset: {selected_dataset}
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Create summary statistics
summary_df = df.describe(include='all').reset_index()
summary_df.columns.name = None
summary_df = summary_df.rename(columns={{'index': 'statistic'}})

# Create correlation matrix for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    # Store correlation matrix as a separate result
    correlation_df = corr_matrix.reset_index()
    correlation_df.columns.name = None
    correlation_df = correlation_df.rename(columns={{'index': 'variable'}})
else:
    correlation_df = None

# Create visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Numeric Column Distribution',
        'Correlation Heatmap',
        'Box Plots',
        'Missing Values'
    ),
    specs=[[{{'type': 'histogram'}}, {{'type': 'heatmap'}}],
           [{{'type': 'box'}}, {{'type': 'bar'}}]]
)

# 1. Numeric Column Distribution (top left)
if len(numeric_cols) > 0:
    for i, col in enumerate(numeric_cols[:3]):  # Show up to 3 columns
        fig.add_trace(
            go.Histogram(x=df[col], name=col, opacity=0.7),
            row=1, col=1
        )

# 2. Correlation Heatmap (top right)
if len(numeric_cols) >= 2:
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0
        ),
        row=1, col=2
    )

# 3. Box Plots (bottom left)
if len(numeric_cols) > 0:
    for i, col in enumerate(numeric_cols[:3]):  # Show up to 3 columns
        fig.add_trace(
            go.Box(y=df[col], name=col),
            row=2, col=1
        )

# 4. Missing Values Bar Chart (bottom right)
missing_data = df.isnull().sum()
if missing_data.any():
    fig.add_trace(
        go.Bar(
            x=missing_data.index,
            y=missing_data.values,
            name='Missing Values'
        ),
        row=2, col=2
    )

# Update layout
fig.update_layout(
    showlegend=True,
    title_text=f'Analysis of {selected_dataset}',
    title_x=0.5,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Store visualization
viz_data = {{
    'type': 'plotly',
    'figure': fig.to_dict()
}}

# Return both DataFrames in a dictionary
result = {{
    'summary': summary_df,
    'correlation': correlation_df
}}
"""
        
        # Build response with clear sections
        response_parts = [
            f"### Analysis Plan for Dataset: {selected_dataset}",
            "\n**Dataset Overview:**",
            f"- Rows: {len(df)}",
            f"- Columns: {len(df.columns)}",
            f"- Numeric columns: {len(numeric_cols)}",
            f"- Categorical columns: {len(categorical_cols)}",
            "\n**Generated Analysis Code:**",
            "```python",
            analysis_code,
            "```",
            "\n**To Execute:**",
            "Run the analysis by typing 'run.' in the chat."
        ]
        
        # Create focused context matching database service pattern
        service_context = ServiceContext(
            source=self.name,
            data={
                'code_id': code_id,
                'dataset_name': selected_dataset,
                'analysis_type': 'summary_with_viz',
                'code': analysis_code
            },
            metadata={
                'dataset_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'numeric_columns': list(numeric_cols)[:5],
                    'categorical_columns': list(categorical_cols)[:5]
                },
                'request': params.get('description', 'general analysis'),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content="\n".join(response_parts),
                message_type="info",
                role="assistant"
            )],
            context=service_context,
            state_updates={'chat_input': ''}
        )

    def _process_multiple_results(self, results: dict, code_id: str, dataset_name: str, store_updates: dict) -> Tuple[list, str, dict]:
        """Process multiple DataFrame results and generate previews."""
        result_ids = []
        previews = []
        
        # Track summary info for context
        result_summary = []
        print("results:", results)
        
        # Create first ID with prefix
        previous_id = None
        
        # Initialize successful_queries_store
        store_updates['successful_queries_store'] = {}
        
        for result_name, df in results.items():
            if df is not None and isinstance(df, pd.DataFrame):
                # Generate ID based on whether we have a previous one
                if previous_id is None:
                    result_id = PreviewIdentifier.create_id(prefix="dataset")
                else:
                    result_id = PreviewIdentifier.create_id(previous_id=previous_id)
                previous_id = result_id  # Update for next iteration
                result_ids.append(result_id)
                
                preview = f"{result_name.title()} Result ({len(df)} rows × {len(df.columns)} columns):\n"
                preview += f"Dataset ID: {result_id}\n"
                preview += f"```\n{df.head().to_string()}\n```"
                previews.append(preview)
                
                # Add to summary
                result_summary.append(f"{result_name}: {len(df)} rows × {len(df.columns)} columns")
                
                store_updates['successful_queries_store'][result_id] = {
                    'type': 'dataframe',
                    'code_id': code_id,
                    'dataframe': df.to_dict('records'),
                    'metadata': {
                        'source_dataset': dataset_name,
                        'result_type': result_name,
                        'execution_time': datetime.now().isoformat(),
                        'rows': len(df),
                        'columns': list(df.columns)
                    }
                }
        
        # Create context data using first DataFrame but including summary
        first_df = next((df for df in results.values() if df is not None), None)
        
        # Build context that summarizes all results
        context_data = {
            'results_summary': {
                name: {
                    'rows': len(df) if df is not None else 0,
                    'columns': list(df.columns) if df is not None else [],
                    'preview': df.head().to_dict('records') if df is not None else {}
                }
                for name, df in results.items()
            },
            'code_id': code_id,
            'dataset_name': dataset_name,
            # Include first DataFrame's data for backward compatibility
            'data': first_df.to_dict('records') if first_df is not None else {}
        }

        # Create service context
        service_context = ServiceContext(
            source=self.name,
            data=context_data,
            metadata={
                'execution_time': datetime.now().isoformat(),
                'result_count': len(results),
                'analysis_prompts': [
                    "Please analyze the results of the code execution:",
                    "- Summarize what each DataFrame contains",
                    "- Note any interesting patterns or relationships between the results",
                    "- Suggest potential next steps for analysis"
                ]
            }
        )

        return result_ids, "\n\n".join(previews), context_data

    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute dataset service request."""
        try:
            # Get command type
            command = params.get('command')
            
            if command == 'info':
                return self._handle_info_request(params, context)
            elif command == 'validate':
                return self._handle_code_validation(params, context)
            elif command == 'execute':
                return self._handle_code_execution(params, context)
            elif command == 'convert':
                return self._handle_dataset_conversion(params, context)
            elif command == 'analysis':
                return self._handle_analysis_request(params, context)
            else:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="Invalid command",
                        message_type="error"
                    )],
                    state_updates={'chat_input': ''}
                )
                
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"{self.name} service error: {str(e)}",
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )

# ... rest of the existing code ... 
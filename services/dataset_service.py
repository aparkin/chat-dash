"""
Dataset service implementation.

This service handles dataset analysis and code execution in the ChatDash application.
It provides a modular interface for dataset operations and integrates LLM capabilities
for intelligent code generation and analysis.

Key Features:
1. Dataset Management:
   - Dataset information and exploration
   - Safe code execution for analysis
   - Dataset creation from analysis results
   - Dataset state management
   - LLM-assisted analysis planning

2. Code Generation:
   - Template-based code generation
   - LLM-powered analysis customization
   - Strict validation of code structure
   - Safe execution environment
   - Result validation and type checking

3. Visualization:
   - Interactive plot generation
   - State management for visualizations
   - Plot configuration validation
   - JSON serialization handling
   - View state persistence

Architecture:
1. Exception Hierarchy:
   - DatasetAnalysisException: Base exception for all dataset-related errors
   - ValidationError: Raised when code validation fails
   - ExecutionError: Raised when code execution fails
   - SecurityError: Raised when code violates security constraints

2. Core Components:
   - CodeValidator: Validates code blocks for safety and correctness
   - CodeExecutor: Executes validated code blocks in a controlled environment
   - DatasetService: Main service class handling all dataset operations

3. Command Types:
   - info: Dataset information requests
   - validate: Code block validation
   - execute: Code execution
   - convert: Dataset conversion
   - analysis: Analysis generation

4. LLM Integration:
   - Uses LLMServiceMixin for code generation
   - Maintains analysis context
   - Handles validation retries
   - Manages token budgets
   - Processes LLM responses

Usage Example:
    ```python
    # Initialize service
    service = DatasetService()
    
    # Handle analysis request
    response = service.execute(
        params={'command': 'analysis', 'description': 'Analyze temperature trends'},
        context={
            'datasets_store': datasets,
            'selected_dataset': 'climate_data',
            'chat_history': history
        }
    )
    
    # Execute generated code
    result = service.execute(
        params={'command': 'execute', 'code_id': 'datasetCode_20240315_123456'},
        context={...}
    )
    ```

Dependencies:
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- plotly: Visualization
- scipy.stats: Statistical functions
- sklearn.preprocessing: Data preprocessing
- ydata_profiling: Dataset profiling

Implementation Notes:
1. Code Generation:
   - Uses template-based approach for consistency
   - Implements strict validation of structure
   - Maintains safe execution environment
   - Validates results and types
   - Manages visualization state

2. Error Handling:
   - Implements comprehensive exception hierarchy
   - Provides detailed error messages
   - Includes validation retry logic
   - Maintains error history for context

3. State Management:
   - Tracks dataset state
   - Manages visualization state
   - Handles execution results
   - Maintains analysis history
   - Preserves user context

4. Security:
   - Validates all code execution
   - Restricts available operations
   - Sanitizes inputs and outputs
   - Manages resource limits
   - Tracks execution history
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
    PreviewIdentifier,
    MessageType
)
from .llm_service import LLMServiceMixin

class DatasetAnalysisException(Exception):
    """Base exception for dataset analysis errors.
    
    This is the parent class for all dataset service specific exceptions.
    It provides a common base for catching and handling dataset-related errors.
    
    Usage:
        try:
            result = service.execute(...)
        except DatasetAnalysisException as e:
            # Handle any dataset-related error
            print(f"Analysis error: {str(e)}")
    """
    pass

class ValidationError(DatasetAnalysisException):
    """Raised when code validation fails.
    
    This exception is raised when code fails to meet safety or structural requirements:
    - Contains blocked operations
    - Uses unauthorized imports
    - Missing required components
    - Invalid code structure
    
    Attributes:
        message: Description of the validation failure
        code: The code that failed validation (if available)
    """
    pass

class ExecutionError(DatasetAnalysisException):
    """Raised when code execution fails.
    
    This exception is raised when:
    - Code execution fails in the controlled environment
    - Required variables are not set
    - Return values are missing or invalid
    - Runtime errors occur during execution
    
    Attributes:
        message: Description of the execution failure
        traceback: Full traceback of the error (if available)
    """
    pass

class SecurityError(DatasetAnalysisException):
    """Raised when code violates security constraints.
    
    This exception is raised when:
    - Code attempts to access restricted operations
    - Code tries to import unauthorized modules
    - Code attempts to modify system state
    - Code violates sandbox restrictions
    
    Attributes:
        message: Description of the security violation
        operation: The specific operation that was blocked
    """
    pass

class CodeValidator:
    """Validates code blocks for safety and correctness.
    
    This class provides a comprehensive validation framework for Python code blocks,
    ensuring they meet safety and structural requirements before execution.
    
    Validation checks:
    1. Safety:
       - No blocked operations (eval, exec, etc.)
       - Only allowed imports
       - No file system or system operations
    
    2. Structure:
       - Required variable initialization
       - Proper function definition
       - Return value format
    
    3. Output Format:
       - result variable must be a dictionary
       - viz_data must contain figure key
    
    Attributes:
        allowed_imports (set[str]): Set of allowed module imports
        blocked_ops (set[str]): Set of blocked operations
        import_pattern (Pattern): Regex for import statements
        blocked_pattern (Pattern): Regex for blocked operations
        result_pattern (Pattern): Regex for result initialization
        viz_data_pattern (Pattern): Regex for viz_data structure
    """
    
    def __init__(self):
        # Allowed module imports
        self.allowed_imports = {
            'pandas', 'numpy', 'plotly', 
            'scipy.stats', 'sklearn.preprocessing'
        }
        
        # Blocked operations/attributes
        self.blocked_ops = {
            'eval', 'exec', 'compile',  # Code execution
            'open', 'file', 'os', 'sys',  # File/system access
            'subprocess', 'import', '__import__',  # System/import operations
            'map'  # Blocked because it can be used for code execution
        }
        
        # Compile regex patterns
        self.import_pattern = re.compile(r'^(?:from|import)\s+([a-zA-Z0-9_.]+)')
        self.blocked_pattern = re.compile(
            r'(?:^|[^a-zA-Z0-9_.])'  # Start of line or non-identifier char
            + r'(?:' + '|'.join(map(re.escape, self.blocked_ops)) + r')'  # Blocked operation
            + r'\s*\('  # Opening parenthesis
        )
        
        # Output format validation patterns - Make more flexible
        self.result_pattern = re.compile(r'(?:result\s*=\s*\{[^}]*\}|result\s*=\s*\{\}|result\[[\'"][^\]]+[\'"]\]\s*=)')
        self.viz_data_pattern = re.compile(r'viz_data\s*=\s*\{[^}]*[\'"]figure[\'"]\s*:')
    
    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code block for safety and correctness.
        
        Performs a series of validation checks on the provided code:
        1. Checks for blocked operations
        2. Validates imports
        3. Verifies required output format
        4. Checks code syntax
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
            - is_valid: True if code passes all validation checks
            - error_message: None if valid, otherwise description of the error
            
        Raises:
            None: All exceptions are caught and returned as error messages
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
            
            # Check for required output format - result dictionary
            if not self.result_pattern.search(code):
                return False, "Code must initialize 'result' variable as a dictionary and populate it with DataFrames"
                
            # Check for required output format - viz_data
            if not self.viz_data_pattern.search(code):
                return False, "Code must set 'viz_data' with 'figure' key"
            
            # Try to compile code to check syntax
            compile(code, '<string>', 'exec')
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class CodeExecutor:
    """Executes validated code blocks safely in a controlled environment.
    
    This class provides a secure execution environment for running user-provided
    code blocks. It manages the execution context, provides necessary libraries,
    and ensures proper isolation of the execution environment.
    
    Features:
    1. Controlled Environment:
       - Pre-imported trusted libraries
       - Restricted global namespace
       - Safe execution context
    
    2. Library Management:
       - Core data libraries (pandas, numpy)
       - Visualization libraries (plotly)
       - Statistical libraries (scipy.stats)
       - Preprocessing tools (sklearn)
    
    3. Result Validation:
       - Type checking of return values
       - Validation of result structure
       - Proper error handling
    
    Attributes:
        validator (CodeValidator): Code validation instance
        globals (dict): Global namespace for code execution
    """
    
    def __init__(self):
        """Initialize the code executor with necessary libraries and validation."""
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
    
    def execute(self, code: str, df: pd.DataFrame, datasets: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Execute code safely in a controlled environment.
        
        This method:
        1. Validates the code using CodeValidator
        2. Sets up a clean execution environment
        3. Executes the code safely
        4. Validates and returns the results
        
        Args:
            code: Python code to execute
            df: DataFrame to make available to the code
            datasets: Dictionary of datasets from store
            
        Returns:
            Tuple[Any, Dict[str, Any]]: (execution_result, metadata)
            - execution_result: The result from the analyze_data function
            - metadata: Dictionary containing execution metadata
            
        Raises:
            ValidationError: If code fails validation
            ExecutionError: If execution fails or returns invalid results
        """
        # First validate the code
        is_valid, error = self.validator.validate(code)
        if not is_valid:
            raise ValidationError(f"Code validation failed: {error}")
            
        try:
            # Create a local namespace for execution
            local_vars = {
                'df': df,
                'datasets': datasets,
                'result': None,
                'viz_data': None
            }
            
            # Execute the code to define the function
            try:
                exec(code, self.globals, local_vars)
            except Exception as e:
                print("\nFunction definition error:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                raise ExecutionError(f"Error defining analyze_data function: {str(e)}")
            
            # Get the analyze_data function from the namespace
            analyze_data = local_vars.get('analyze_data')
            if not analyze_data:
                raise ExecutionError("analyze_data function not found in code")
            
            # Execute the function and get results
            try:
                result, viz_data = analyze_data(datasets)
            except Exception as e:
                print("\nFunction execution error:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                
                raise ExecutionError(f"Error executing analyze_data function: {str(e)}")
            
            # Verify we got both values
            if result is None:
                raise ExecutionError("analyze_data function did not return a result")
            if viz_data is None:
                raise ExecutionError("analyze_data function did not return viz_data")
            
            # Collect metadata about the execution
            metadata = {
                'execution_time': datetime.now().isoformat(),
                'code_size': len(code),
                'has_viz': bool(viz_data and isinstance(viz_data, dict) and 'figure' in viz_data)
            }
            
            # If there's visualization data, include it in metadata
            if metadata['has_viz']:
                metadata['viz_data'] = viz_data
            
            return result, metadata
                
        except Exception as e:
            print(f"\nExecution error details: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Local variables: {list(local_vars.keys())}")
            if 'result' in local_vars:
                print(f"Result type: {type(local_vars['result'])}")
            if 'viz_data' in local_vars:
                print(f"viz_data type: {type(local_vars['viz_data'])}")
            if 'analyze_data' in local_vars:
                print(f"analyze_data type: {type(local_vars['analyze_data'])}")
            print("Traceback:")
            print(traceback.format_exc())
            raise ExecutionError(f"Code execution failed: {str(e)}")

class DatasetService(ChatService, LLMServiceMixin):
    """Service for dataset analysis and code execution."""
    
    def __init__(self):
        ChatService.__init__(self, "dataset")
        LLMServiceMixin.__init__(self, "dataset")
        
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
        # Store original message for analysis descriptions
        original_message = message.strip()
        print(f"\nParsing dataset request: '{original_message}'")
        # Use lowercase only for command detection
        message = message.lower().strip()

        
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
            
        # Check for analysis request - use original message to preserve case
        analysis_match = re.match(r'^analysis:\s*(.+)$', message)
        if analysis_match:
            # Get the description from the original message to preserve case
            description_start = original_message.find(':') + 1
            return {
                'command': 'analysis',
                'description': original_message[description_start:].strip()
            }
            
        # Check for Python code block
        if self.code_block_re.search(message):
            return {
                'command': 'validate',
                'code': original_message  # Use original message to preserve case in code
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
                test_datasets = {'test': {'df': df.to_dict('records')}}
                result, metadata = self.executor.execute("result = df.describe()", df, test_datasets)
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
            )]
        )

    def _handle_info_request(self, params: dict, context: dict) -> ServiceResponse:
        """Handle dataset information requests."""
        datasets = context.get('datasets_store', {})

        if not datasets:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No datasets are currently loaded.",
                    message_type=MessageType.ERROR,
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
            
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="\n".join(parts),
                    message_type=MessageType.RESULT,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
            
        else:
            # Show detailed info for specific dataset
            if target not in datasets:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Dataset '{target}' not found.",
                        message_type=MessageType.ERROR,
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
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="\n".join(parts),
                    message_type=MessageType.RESULT,
                    role="assistant"
                )],
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
                    message_type=MessageType.ERROR,
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
                    message_type=MessageType.ERROR,
                    role="assistant"
                ))
            else:
                responses.append(ServiceMessage(
                    service=self.name,
                    content="✓ Code validation successful",
                    message_type=MessageType.RESULT,
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
            print("\nExecuting code...")
            result, metadata = self.executor.execute(code, df, context['datasets_store'])
            
            print("\nExecution complete")
            print("Result type:", type(result))
            print("Has visualization:", metadata.get('has_viz', False))
            
            # Process results and build response components
            store_updates = {}
            state_updates = {'chat_input': ''}
            messages = []
            
            # Handle visualization if present
            if metadata.get('has_viz') and metadata.get('viz_data'):
                viz_data = metadata['viz_data']
                if isinstance(viz_data, dict) and 'figure' in viz_data:
                    state_updates['active_tab'] = 'tab-viz'
                    state_updates['viz_state'] = self._process_visualization(viz_data, selected_dataset, df, context)
                    if state_updates['viz_state']:
                        messages.append(ServiceMessage(
                            service=self.name,
                            content="✓ Visualization created and available in visualization tab",
                            message_type=MessageType.INFO,
                            role="assistant"
                        ))

            # Process results using new unified method
            result_ids, preview_text = self._process_results(result, code_id, selected_dataset, store_updates)
            main_result_id = result_ids[0] if result_ids else None
            
            # Generate summary of results
            summary = self.summarize(preview_text, context['chat_history'])
            
            # Build execution response
            if preview_text:  # Only add result message if we have content
                messages.extend([
                    ServiceMessage(
                        service=self.name,
                        content=self._build_execution_response(
                            preview_text,
                            main_result_id,
                            bool(state_updates.get('viz_state'))
                        ),
                        message_type=MessageType.RESULT,
                        role="assistant"
                    ),
                    ServiceMessage(
                        service=self.name,
                        content=f"### Analysis Summary\n\n{summary}",
                        message_type=MessageType.SUMMARY,
                        role="assistant"
                    )
                ])
            else:
                # Add a default message if no preview text
                messages.append(ServiceMessage(
                    service=self.name,
                    content=self._build_execution_response(
                        "Code executed successfully but produced no preview output.",
                        main_result_id,
                        bool(state_updates.get('viz_state'))
                    ),
                    message_type=MessageType.RESULT,
                    role="assistant"
                ))
            
            print("\nResponse built")
            print("Number of messages:", len(messages))
            print("Has visualization:", bool(state_updates.get('viz_state')))

            return ServiceResponse(
                messages=messages,
                store_updates=store_updates,
                state_updates=state_updates
            )

        except Exception as e:
            print(f"\nError during execution: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            error_msg = f"Dataset service error in code execution: {str(e)}"
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=error_msg,
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )

    def _validate_execution_environment(self, params: dict, context: dict) -> Union[ServiceResponse, Tuple[pd.DataFrame, str, str, str]]:
        """Validate execution environment and extract required components."""
        print("\n=== Debug: Validation Environment ===")
        print(f"Validation parameters: {params}")
        
        # Check for selected dataset
        selected_dataset = context.get('selected_dataset')
        datasets = context.get('datasets_store', {})
        
        if not datasets:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Dataset service error in validation: No datasets are currently loaded.",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        if not selected_dataset:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset is selected. Please select a dataset from the list before running an analysis.",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get code to execute
        code_id = params.get('code_id')
        print(f"\nLooking for code with ID: {code_id}")
        code = self._find_code_block(context['chat_history'], code_id)
        
        if not code:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No Python code block found in recent chat history.",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        print("\nFound code block:")
        print("First 200 characters:")
        print(code[:200])
        
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
            
        # Convert numpy arrays and pandas types to standard Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Int64Dtype):
                return 'Int64'
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            # Add handling for plotly Interval objects and other special types
            elif hasattr(obj, 'to_plotly_json'):
                return obj.to_plotly_json()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj
            
        # Convert the entire figure
        figure = convert_to_serializable(figure)
            
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

    def _process_results(self, result: Any, code_id: str, dataset_name: str, store_updates: dict) -> Tuple[List[str], str]:
        """Process execution results and prepare preview.
        
        Handles both single and multiple results, providing consistent ID generation
        and preview formatting.
        
        Args:
            result: Single result or dictionary of results
            code_id: ID of the code that generated the results
            dataset_name: Name of the source dataset
            store_updates: Dictionary to store updates for the successful queries store
            
        Returns:
            Tuple[List[str], str]: (list of result IDs, formatted preview text)
        """
        result_ids = []
        preview_parts = []
        
        # Initialize store if needed
        store_updates['successful_queries_store'] = {}
        
        # Handle dictionary of results
        if isinstance(result, dict) and all(isinstance(v, (pd.DataFrame, type(None))) for v in result.values()):
            # Create first ID with prefix
            previous_id = None
            
            for result_name, df in result.items():
                if df is not None and isinstance(df, pd.DataFrame):
                    # Generate ID based on whether we have a previous one
                    if previous_id is None:
                        result_id = PreviewIdentifier.create_id(prefix="dataset")
                    else:
                        result_id = PreviewIdentifier.create_id(previous_id=previous_id)
                    previous_id = result_id
                    result_ids.append(result_id)
                    
                    # Format preview
                    preview_parts.append(f"\n**{result_name}** ({len(df)} rows × {len(df.columns)} columns)")
                    preview_parts.append(f"Dataset ID: {result_id}")
                    preview_parts.append(f"```\n{df.head().to_string()}\n```")
                    
                    # Store result
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
        
        # Handle single result
        else:
            result_id = PreviewIdentifier.create_id(prefix="dataset")
            result_ids.append(result_id)
            
            if isinstance(result, pd.DataFrame):
                preview_parts.append(f"DataFrame Result ({len(result)} rows × {len(result.columns)} columns)")
                preview_parts.append(f"Dataset ID: {result_id}")
                preview_parts.append(f"```\n{result.head().to_string()}\n```")
                
                store_updates['successful_queries_store'][result_id] = {
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
                
            elif isinstance(result, pd.Series):
                preview_parts.append(f"Series Result ({len(result)} elements)")
                preview_parts.append(f"Dataset ID: {result_id}")
                preview_parts.append(f"```\n{result.head().to_string()}\n```")
                
                store_updates['successful_queries_store'][result_id] = {
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
                
            else:
                preview_parts.append(f"Result:")
                preview_parts.append(f"Dataset ID: {result_id}")
                preview_parts.append(f"```\n{str(result)[:1000]}\n```")
                
                store_updates['successful_queries_store'][result_id] = {
                    'type': 'other',
                    'code_id': code_id,
                    'result': str(result),
                    'metadata': {
                        'source_dataset': dataset_name,
                        'execution_time': datetime.now().isoformat(),
                        'result_type': type(result).__name__
                    }
                }
        
        return result_ids, "\n".join(preview_parts)

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
        
        most_recent_block = None
        most_recent_id = None
        
        for msg in reversed(chat_history):
            if msg['role'] == 'assistant' and '```python' in msg['content'].lower():
                
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
                        
                        # Store the most recent block we've seen
                        if most_recent_block is None:
                            most_recent_block = block
                            most_recent_id = found_id
                            print("Storing as most recent block")
                        
                        if code_id:
                            if found_id.lower() == code_id.lower():  # Case-insensitive comparison
                                print(f"Matched requested ID: {code_id}")
                                print("Block content preview:")
                                print(block[:200])
                                return block
                        
                # If we've found a message with Python blocks but haven't returned,
                # and we're looking for the most recent block, return what we found
                if not code_id and most_recent_block:
                    print("\nNo specific ID requested, returning most recent block:")
                    print(f"ID: {most_recent_id}")
                    print("Block content preview:")
                    print(most_recent_block[:200])
                    return most_recent_block
                            
        if code_id:
            print(f"No matching code block found for ID: {code_id}")
        else:
            print("No Python code blocks found in recent history")
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
                    message_type=MessageType.ERROR,
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
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
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
            
            response = ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Code results converted to dataset '{dataset_id}'

- Rows: {len(df)}
- Columns: {', '.join(df.columns)}
- Source: Code {dataset_id}
- Type: {stored['metadata'].get('result_type', 'dataframe')}""",
                    message_type=MessageType.RESULT,
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
            
            return response
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"{self.name} service error: ❌ Dataset conversion failed: {str(e)}",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )

    def _get_test_analysis_code(self, code_id: str, selected_dataset: str) -> str:
        """Get the standard test analysis code template.
        This is our reference implementation that demonstrates the expected format.
        """
        return f'''# Code ID: {code_id}
# Analysis for dataset: {selected_dataset}

def analyze_data(datasets):
    """Analyze dataset and generate summary statistics with visualizations."""
    # Load the data
    df = pd.DataFrame(datasets['{selected_dataset}']['df'])
    
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

    # Store results
    result = {{
        'summary': summary_df,
        'correlation': correlation_df
    }}

    # Store visualization
    viz_data = {{
        'type': 'plotly',
        'figure': fig.to_dict()
    }}
    
    return result, viz_data

# Execute analysis
result, viz_data = analyze_data(datasets)'''

    def process_message(self, message: str, chat_history: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Process a message using the service's LLM for code generation.
        
        This is the central method for LLM-based code generation, used by _handle_analysis_request
        and other methods that need to generate code through the LLM.
        
        Args:
            message: The user's message to process
            chat_history: List of previous chat messages
            context: Optional dictionary containing additional context like selected dataset
            
        Returns:
            str: The LLM's response containing generated code
        """
        print("\n=== Analysis Generation Debug ===")
        
        # Get datasets store from chat history context
        datasets_store = {}
        for msg in reversed(chat_history):
            if msg.get('store_updates', {}).get('datasets_store'):
                datasets_store = msg['store_updates']['datasets_store']
                break
        
        print(f"Found datasets: {list(datasets_store.keys())}")
        
        # Get selected dataset from context or None if not available
        selected_dataset = context.get('selected_dataset') if context else None
        
        # Prepare context with dataset information and get token limits
        system_prompt, context_messages, limits = self._prepare_llm_context(
            message, chat_history, datasets_store, selected_dataset
        )
        
        print(f"Token limits: {limits}")
        
        # Add the current message
        context_messages.append({
            "role": "user",
            "content": message
        })
        
        # Initialize validation state
        max_retries = 2
        retry_count = 0
        validation_history = []
        
        while retry_count < max_retries:
            try:
                print(f"\nAttempt {retry_count + 1} of {max_retries}")
                
                # Get LLM response
                response = self._call_llm(context_messages, system_prompt)
                print("\nGot LLM response, length:", len(response))
                
                # Generate a proper code ID before adding to blocks
                code_id = PreviewIdentifier.create_id(prefix="datasetCode")
                print(f"Generated code ID: {code_id}")
                
                # Replace template ID placeholder with actual ID
                response = response.replace("{ID}", code_id)
                
                # Add proper code IDs to any additional code blocks
                response = self.add_ids_to_blocks(response)
                
                # Extract code blocks
                code_blocks = self.detect_content_blocks(response)
                if not code_blocks:
                    raise ValueError("Response must include Python code block")
                
                print(f"Found {len(code_blocks)} code blocks")
                
                # Process each code block
                for code, _, _ in code_blocks:
                    print("\nValidating code block:")
                    print("First 200 characters:")
                    print(code[:200])
                    print("\nLast 200 characters:")
                    print(code[-200:])
                    
                    # Check code size
                    code_tokens = self.count_tokens(code)
                    print(f"Code tokens: {code_tokens}/{limits['code_output']}")
                    if code_tokens > limits['code_output']:
                        raise ValueError(f"Code too long ({code_tokens} tokens)")
                    
                    # Validate the code structure
                    is_valid_structure, structure_error = self._validate_code_structure(code)
                    if not is_valid_structure:
                        raise ValueError(f"Code structure validation failed: {structure_error}")
                    
                    # Validate the code safety
                    is_valid, error = self.executor.validator.validate(code)
                    if not is_valid:
                        raise ValueError(f"Code validation failed: {error}")
                    
                    # Verify proper dataset access pattern
                    if "df = " in code and "datasets[" not in code:
                        raise ValueError("Code must use datasets dictionary to access data")
                    print("Dataset access pattern validation passed")
                
                # If we get here, all validation passed
                print("\nAll validations passed, returning response")
                return response
                
            except ValueError as e:
                error_msg = str(e)
                print(f"\nValidation error: {error_msg}")
                
                # Check if we've seen this error before
                if error_msg in validation_history:
                    retry_count += 1  # Increment retry count for repeated errors
                validation_history.append(error_msg)
                
                # Add the failed response and error to context
                if len(context_messages) > 3:  # Keep context size manageable
                    context_messages = context_messages[:3] + context_messages[-2:]
                    
                context_messages.extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Error: {error_msg}. Please fix and try again."}
                ])
                
                retry_count += 1
                if retry_count >= max_retries:
                    print("\nFailed to generate valid code after maximum retries")
                    print("Last error:", error_msg)
                    print("\nLast generated code:")
                    for code, _, _ in code_blocks:
                        print("\n--- Code Block ---")
                        print(code)
                        print("------------------")
                    return f"""Failed to generate valid code after {max_retries} attempts.
Last error: {error_msg}
Please try rephrasing your request or simplifying the analysis."""
                    
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                print("\nLast generated code:")
                if 'code_blocks' in locals():
                    for code, _, _ in code_blocks:
                        print("\n--- Code Block ---")
                        print(code)
                        print("------------------")
                return f"Unexpected error in code generation: {str(e)}"
        
        return response

    def _handle_analysis_request(self, params: dict, context: dict) -> ServiceResponse:
        """Handle analysis requests with code generation.
        
        This method supports two paths:
        1. "analysis: test" - Uses our reference implementation with a standard template
        2. "analysis: {description}" - Uses LLM to generate custom analysis with full context
        """
        # Validate dataset selection
        selected_dataset = context.get('selected_dataset')
        datasets = context.get('datasets_store', {})
        
        if not datasets or not selected_dataset:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset is selected. Please select a dataset from the list before running an analysis.",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
        
        try:
            # Get analysis description
            description = params.get('description', '').strip()
            print(f"\nAnalysis description: '{description}'")
            print(f"Length: {len(description)}")
            print(f"Is test? {description.lower() == 'test'}")
            
            if not description:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="Please provide an analysis description.",
                        message_type=MessageType.ERROR,
                        role="assistant"
                    )],
                    state_updates={'chat_input': ''}
                )
            
            # Get dataset info for context
            df = pd.DataFrame(datasets[selected_dataset]['df'])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Generate code ID
            code_id = PreviewIdentifier.create_id(prefix="datasetCode")
            
            # Check if this is a test request
            if description.lower() == 'test':
                print("\nUsing test analysis code")
                analysis_code = self._get_test_analysis_code(code_id, selected_dataset)
                # Create a standard template response for test analysis
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
                    "\n**Expected Results:**",
                    "1. Summary statistics for all columns",
                    "2. Correlation matrix for numeric columns",
                    "3. Visualizations showing distributions and relationships",
                    "\n**To Execute:**",
                    "Run the analysis by typing 'run.' in the chat."
                ]
                response_content = "\n".join(response_parts)
            else:
                print("\nGenerating custom analysis code")
                try:
                    # Use process_message for LLM code generation
                    print("\nCalling LLM for code generation...")
                    response = self.process_message(
                        f"Please generate analysis code for: {description}",
                        context['chat_history'],
                        context
                    )
                    print("\nLLM Response received, length:", len(response))
                    
                    # Validate that response contains code block
                    code_blocks = self.detect_content_blocks(response)
                    if not code_blocks:
                        print("\nNo code blocks found in response:")
                        print(response)
                        raise ValueError("No code block found in LLM response")
                    
                    print(f"\nFound {len(code_blocks)} code blocks")
                    for i, (code, _, _) in enumerate(code_blocks):
                        print(f"\nCode Block {i+1}:")
                        print(code)
                    
                    # Keep the full LLM response as it contains valuable context
                    response_content = response
                        
                except Exception as e:
                    print(f"\nLLM code generation failed: {str(e)}")
                    print("\nFailed code generation attempt:")
                    if 'response' in locals():
                        print("\nComplete LLM Response:")
                        print("-------------------")
                        print(response)
                        print("-------------------")
                        if 'code_blocks' in locals() and code_blocks:
                            print("\nExtracted code blocks")
                        else:
                            print("No code blocks were extracted from the response")
                    else:
                        print("No response was received from LLM")
                    print("\nError traceback:")
                    print(traceback.format_exc())
                    print("\nFalling back to test analysis")
                    
                    # Fall back to test analysis with standard template
                    analysis_code = self._get_test_analysis_code(code_id, selected_dataset)
                    response_parts = [
                        f"### Analysis Plan for Dataset: {selected_dataset}",
                        "\n**Note:** Custom analysis generation failed, falling back to standard analysis.",
                        f"\nError: {str(e)}",
                        "\nFailed code generation attempt:",
                        "```",
                        "Complete LLM Response:" if 'response' in locals() else "No response received",
                        response if 'response' in locals() else "",
                        "```",
                        "\n**Dataset Overview:**",
                        f"- Rows: {len(df)}",
                        f"- Columns: {len(df.columns)}",
                        f"- Numeric columns: {len(numeric_cols)}",
                        f"- Categorical columns: {len(categorical_cols)}",
                        "\n**Generated Analysis Code:**",
                        "```python",
                        analysis_code,
                        "```",
                        "\n**Expected Results:**",
                        "1. Summary statistics for all columns",
                        "2. Correlation matrix for numeric columns",
                        "3. Visualizations showing distributions and relationships",
                        "\n**To Execute:**",
                        "Run the analysis by typing 'run.' in the chat."
                    ]
                    response_content = "\n".join(response_parts)
                    print("#######\n", response_content, "\n#######")
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=response_content.strip(),
                    message_type=MessageType.RESULT,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error generating analysis code: {str(e)}",
                    message_type=MessageType.ERROR,
                    role="assistant"
                )],
                state_updates={'chat_input': ''}
            )

    def _calculate_context_limits(self, system_prompt: str) -> Dict[str, int]:
        """Calculate safe token limits for different components.
        
        Manages both input context window and expected output size to ensure:
        1. Input (prompt + history + message) stays within model limits
        2. Output can be used in future context
        3. Space available for validation iterations
        
        Args:
            system_prompt: The base system prompt
            
        Returns:
            Dict with token limits for different components
        """
        # Model context window
        MAX_CONTEXT_TOKENS = 8192
        
        # Calculate system prompt tokens (includes template, requirements)
        system_tokens = self.count_tokens(system_prompt)
        
        # Token allocations
        allocations = {
            'system_prompt': system_tokens,
            'current_message': 1000,    # Increased for user message
            'validation_exchange': 1500,  # Increased for validation
            'code_output': 3000,       # Increased for code generation
            'safety_margin': 1000       # Increased safety margin
        }
        
        # Calculate remaining space for history
        total_reserved = sum(allocations.values())
        available_for_history = MAX_CONTEXT_TOKENS - total_reserved
        
        # Update allocations with history limit
        allocations['history'] = max(0, available_for_history)
        
        # Add total and remaining for reference
        allocations['total_available'] = MAX_CONTEXT_TOKENS
        allocations['remaining'] = MAX_CONTEXT_TOKENS - total_reserved
        
        return allocations

    def _filter_history_by_tokens(self, chat_history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Filter chat history to fit within token limit while preserving context.
        
        Prioritizes:
        1. Most recent messages
        2. Messages from this service
        3. Messages with code blocks
        
        Args:
            chat_history: Full chat history
            max_tokens: Maximum tokens to use
            
        Returns:
            Filtered chat history
        """
        filtered_messages = []
        token_count = 0
        
        # First pass: get recent service messages and code blocks
        for msg in reversed(chat_history):
            msg_tokens = self.count_tokens(msg.get('content', ''))
            
            # Check if message contains code
            has_code = '```python' in msg.get('content', '').lower()
            is_service_msg = msg.get('service') == self.name
            
            # Prioritize messages with code or from our service
            if (has_code or is_service_msg) and token_count + msg_tokens <= max_tokens:
                filtered_messages.insert(0, msg)
                token_count += msg_tokens
        
        # Second pass: add other recent context if space remains
        remaining_tokens = max_tokens - token_count
        if remaining_tokens > 0:
            for msg in reversed(chat_history):
                if msg not in filtered_messages:  # Skip already included messages
                    msg_tokens = self.count_tokens(msg.get('content', ''))
                    if token_count + msg_tokens <= max_tokens:
                        filtered_messages.insert(0, msg)
                        token_count += msg_tokens
        
        return filtered_messages

    CODE_TEMPLATE = '''# The line below MUST remain exactly as is - do not modify the ID placeholder
# Code ID: {{ID}}

# Import statements - DO NOT move these inside the function
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn

def analyze_data(datasets):
    """Analyze datasets and return results with visualization.
    
    Args:
        datasets: Dictionary of datasets from store
        
    Returns:
        tuple: (result_dict, viz_dict) where:
            - result_dict: Dict[str, pd.DataFrame] containing analysis results
            - viz_dict: Dict with 'figure' key containing Plotly figure
    """
    # Analysis implementation
    {implementation}
    
    # Ensure results are in correct format
    if not isinstance(result, dict) or not all(isinstance(df, pd.DataFrame) for df in result.values()):
        raise ValueError("Result must be a dictionary of DataFrames")
    if not isinstance(viz_data, dict) or 'figure' not in viz_data:
        raise ValueError("viz_data must be a dictionary with 'figure' key")
    
    return result, viz_data

# Execute analysis
result, viz_data = analyze_data(datasets)'''

    def _prepare_llm_context(self, message: str, chat_history: List[Dict[str, Any]], datasets_store: Dict, selected_dataset: str) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Prepare context for LLM including dataset information and validation history."""
        print("\n=== Preparing LLM Context ===")
        print(f"Selected dataset: {selected_dataset}")
        
        # Build rich dataset context
        dataset_info = self._build_dataset_context(datasets_store)
        print(f"Built dataset context")

        # Get validation history from recent messages
        validation_history = []
        for msg in reversed(chat_history[-10:]):
            if msg.get('service') == self.name and msg.get('message_type') == 'error':
                if 'validation failed' in msg.get('content', '').lower():
                    validation_history.append(msg['content'])
        
        print(f"Found {len(validation_history)} validation errors in history")
        
        # Generate a code ID for this request
        code_id = PreviewIdentifier.create_id(prefix="datasetCode")
        print(f"Generated code ID: {code_id}")
        
        # Create system prompt with context
        system_prompt = f"""You are a data analysis code generator that creates Python code for dataset analysis.

ANALYSIS REQUEST: [This is the primary user request that you must fulfill]
{message}

TARGET DATASET: '{selected_dataset}'

DATASET INFORMATION:
{dataset_info}

Code Template (DO NOT MODIFY ANY PART OF THIS TEMPLATE):
```python
{self.CODE_TEMPLATE.format(ID=code_id, implementation="# Your implementation here")}
```
Recent validation issues to avoid:
{chr(10).join(validation_history) if validation_history else "No recent validation issues"}

Goal: Produce a robust code to produce a dictionary of informative analysis results in dataframes and/or a rich informative plotly fig. 
You must produce at least one dataframe or one figure.

CORE CONSTRAINTS (ABSOLUTELY REQUIRED):
1. Code Structure:
   - MUST use the provided template WITHOUT ANY MODIFICATIONS
   - MUST keep the '# Code ID: {code_id}' line EXACTLY as shown (no curly braces)
   - MUST implement ALL logic inside the analyze_data function
   - MUST access data using the datasets dictionary and selected dataset:
      CORRECT:   df = pd.DataFrame(datasets['{selected_dataset}']['df'])
      INCORRECT: df = ... or using df directly  

2. Results Structure:
   - MUST return a dictionary named 'result' containing ONLY DataFrames (or empty if only a visualization is produced)
   - EVERY DataFrame in results MUST:
     * Be reset using reset_index()
     * Have descriptive column names
     * Be properly formatted for reuse
   Example format (REQUIRED):
   result = {{
       'summary_stats': df.describe().reset_index(),
       'analysis_results': analysis_df.reset_index(),
       # Add more result DataFrames as needed
   }}

3. Visualization Structure (if needed):
   - MUST use EXACTLY this format: 
   viz_data = {{
       'type': 'plotly',
       'figure': fig.to_dict()  # MUST use to_dict()
   }}
   - MUST use the plotly library to create the figure
   - However, it is not necessary to create a figure if not needed. In this case the viz_data should be a dictionary with a 'type' and a 'figure' key set to 'none'.
   - Use clear titles and labels
   - Include error bars where applicable
   - Use appropriate plot types for data types
   - when feasible, use plots as close as possible to what the user asks for
   - ALWAYS convert Plotly figures to dictionary using fig.to_dict()
   - Set appropriate figure layout:
    ```python
    fig.update_layout(
        title_x=0.5,  # Center title
        showlegend=True,  # Show legend when multiple traces
        paper_bgcolor='white',  # White background
        plot_bgcolor='white'    # White plot area
    )
    ```
    - LEAVE OUT vertical and horizontal space directives when you updating layout or defining the plot or multiplot. 
    - Do not set the height and width of the figure.
    - ALWAYS convert Plotly figures to dictionary using fig.to_dict()
    - If using a multiplot, ENSURE the specs argument matches the type of plots you are making and putting in each column and row.
        For example: 
        - spec='xy' for scatter, histogram, and bar charts
        - spec='heatmap' for heatmap
        - spec='box' for box and violin
        - spec='domain' for pie
        - spec='surface' for surface
        - spec='chloropleth' for chloropleth
        - spec='treemap' for treemap
        - spec='funnel' for funnel
        - spec='candlestick' for candlestick
        - spec='contour' for contour
        - spec='scattergeo' for scattergeo
    - Include hover information for better interactivity
    - Set appropriate margins and spacing
    - For multiple subplots, use descriptive subplot titles 

4. ABSOLUTELY FORBIDDEN Operations: These are security risks and will result in an error. 
   - eval, exec, compile (code execution)
   - open, file, os, sys (file/system access)
   - subprocess, import, __import__ (system/import operations)
   - map (can be used for code execution)
   - ANY file operations or system access
   - ANY dynamic code execution
   - ANY additional imports beyond pre-imported libraries

ANALYSIS REQUIREMENTS:
1. Focus on the selected dataset: '{selected_dataset}'
2. Make sure all code is robust to missing data through proper coercion or handling if necessary.  
3. Validate data before operations:
   - Check column existence
   - Verify non-empty DataFrames/Series before .iloc/.index
   - Validate categorical operations with value_counts()
4. If the user has specified non-existent columns do not generate code but help the user correct their request.

DATA VALIDATION REQUIREMENTS:
1. ONLY use columns that are listed above in the dataset information
2. AVOID using direct numeric indexing - always use column names
3. Column names are CASE SENSITIVE - use exact case as shown in dataset information
4. Always verify column existence before use:
   ```python
   if 'Column_Name' not in df.columns:  # Use exact case
       raise ValueError("Required column 'Column_Name' not found in dataset")
   ```

AVAILABLE LIBRARIES (pre-imported, DO NOT import others):
- pandas (pd): Data manipulation and analysis
- numpy (np): Numerical operations
- plotly.express (px): High-level plotting
- plotly.graph_objects (go): Low-level plotting
- plotly.subplots (make_subplots): Multiple plots
- scipy.stats: Statistical functions
- sklearn.preprocessing: Data preprocessing
- sklearn
- CRITICAL- These are preimported and DO NOT import others. You are not allowed to import any other libraries. 

RESPONSE FORMAT:
1. Analysis Plan:
   - Verify available columns from dataset info
   - List specific steps to accomplish the task
   - Identify required operations

2. Implementation:
   - Complete code block using template
   - Verify all columns exist in dataset info
   - Ensure all requirements are met
   - Confirm results dictionary format

3. Results Explanation:
   - Describe each DataFrame in results
   - Explain visualization choices
   - Note potential insights

4. Execution Instructions:
   "To execute this analysis, type 'run.' in the chat."

Use chain-of-thought reasoning to determine the best way to approach the analysis while meeting all the requirements.
CRITICAL REMINDERS:
- ONLY use columns listed in dataset info
- EVERY result value in the results dictionary must be a DataFrame
- ALL DataFrames must use reset_index()
- NO raw statistical results
- NO additional imports
- EXACT template compliance
- CORRECT subplot specs for plot types
"""

        print("System prompt prepared, length:", len(system_prompt))
        
        # Calculate safe token limits
        limits = self._calculate_context_limits(system_prompt)
        print(f"Token limits calculated: {limits}")
        
        # Prepare context messages with token limit
        context_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add relevant history within token limit
        token_count = self.count_tokens(system_prompt)
        history_messages = []
        
        # First add most recent messages up to token limit
        for msg in reversed(chat_history[-10:]):
            msg_tokens = self.count_tokens(msg.get('content', ''))
            if token_count + msg_tokens > limits['history']:
                break
                
            if msg.get('service') == self.name:
                history_messages.insert(0, {
                    "role": "assistant",
                    "content": msg['content']
                })
            else:
                history_messages.insert(0, {
                    "role": msg['role'],
                    "content": msg['content']
                })
            token_count += msg_tokens
        
        # Create final context messages
        context_messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        context_messages.extend(history_messages)
        
        # Verify total token count
        total_tokens = sum(self.count_tokens(msg['content']) for msg in context_messages)
        print(f"Total context tokens: {total_tokens}/{limits['total_available']}")
        
        if total_tokens > limits['total_available']:
            print("Warning: Context too large, truncating history")
            # Keep system prompt and most recent messages
            context_messages = [context_messages[0]] + context_messages[-2:]
            total_tokens = sum(self.count_tokens(msg['content']) for msg in context_messages)
            print(f"Truncated context tokens: {total_tokens}/{limits['total_available']}")
        
        print(f"Added {len(context_messages)-1} history messages")
        return system_prompt, context_messages, limits

    def _validate_code_structure(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate that code follows the required template structure.
        
        Args:
            code: The code block to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Check for required template elements
            if not code.startswith("# Code ID:"):
                return False, "Code must start with '# Code ID:' line"
                
            # Check for analyze_data function
            if "def analyze_data(datasets):" not in code:
                return False, "Code must contain analyze_data function definition"
                
            # Check for imports
            if any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')):
                return False, "Code should not contain import statements - they are in the template"
                
            # Check for function definitions
            function_defs = [line for line in code.split('\n') if line.strip().startswith('def ')]
            if len(function_defs) > 1:  # Only analyze_data allowed
                return False, "Code should not contain additional function definitions"
                
            # Check for required return structure
            if "return result, viz_data" not in code:
                return False, "Code must return result and viz_data"
                
            return True, None
            
        except Exception as e:
            return False, f"Structure validation failed: {str(e)}"

    def summarize(self, content: str, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of analysis results.
        
        Args:
            content: The content to summarize (analysis results)
            chat_history: List of previous chat messages
            
        Returns:
            str: The generated summary
        """
        # Get relevant context from chat history
        context = self._filter_history_by_tokens(chat_history, max_tokens=4000)
        
        # Create system prompt for result summarization
        system_prompt = """You are a data analysis assistant that helps summarize and interpret analysis results.
        Your task is to:
        1. Understand the analysis results provided
        2. Extract key findings and insights
        3. Suggest potential next steps or follow-up analyses
        
        Format your response as:
        1. Key findings (2-3 bullet points)
        2. Detailed interpretation
        3. Suggested next steps
        4. Any specific information you can use to link these results to general biological and environmental knowledge as follows:
        4a. If there is latitude and longitude data, you can use it to link to specific geographic or environmental conditions or species distributions.
        4b. If there is date data, you can use it to link to seasonal or climatic conditions.
        4c. If there is depth data, you can use it to link to hydrographic or oceanographic conditions.
        4d. If there chemical concentration data, you can use it to link to known environmental distributions or toxicological effects.
        4e. If there is species data, you can use it to link to known species distributions or ecological interactions and functions. 
        4f. If there is temperature data, you can use it to link to known temperature distributions and effects on species and ecosystems.
        Focus on making the results accessible and actionable for the user.
        """
        
        # Convert context messages
        context_messages = [
            {
                "role": "system" if msg.get("role") == "system" else "user",
                "content": msg.get("content", "")
            }
            for msg in context
        ]
        
        # Add the content to summarize
        context_messages.append({
            "role": "user",
            "content": f"Please summarize these analysis results:\n\n{content}"
        })
        
        # Get LLM response
        return self._call_llm(context_messages, system_prompt)

    def _build_dataset_context(self, datasets_store: Dict) -> str:
        """Build rich context information about available datasets.
        
        Args:
            datasets_store: Dictionary of available datasets
            
        Returns:
            str: Formatted dataset information including structure and statistics
        """
        context_parts = []
        
        for name, data in datasets_store.items():
            df = pd.DataFrame(data['df'])
            
            # Basic dataset info
            parts = [
                f"\nDataset: {name}",
                f"Rows: {len(df)}",
                f"Columns: {len(df.columns)}"
            ]
            
            # Column information
            column_info = []
            for col in df.columns:
                stats = []
                dtype = df[col].dtype
                stats.append(f"type: {dtype}")
                
                # Add type-specific statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats.extend([
                        f"range: [{df[col].min():.2f} to {df[col].max():.2f}]",
                        f"missing: {df[col].isna().sum()}"
                    ])
                else:
                    n_unique = df[col].nunique()
                    stats.extend([
                        f"unique values: {n_unique}",
                        f"missing: {df[col].isna().sum()}"
                    ])
                
                column_info.append(f"  - {col} ({', '.join(stats)})")
            
            parts.append("\nColumns:")
            parts.extend(column_info)
            
            context_parts.append("\n".join(parts))
        
        return "\n".join(context_parts)

    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute dataset service request."""
        try:
            # Store context for use in _call_llm
            self.context = context

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
                        message_type=MessageType.ERROR
                    )],
                    state_updates={'chat_input': ''}
                )
                
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"{self.name} service error: {str(e)}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )

    def get_help_text(self) -> str:
        """Get help text for dataset service commands."""
        return """
📈 **Dataset Operations**
- View dataset information: `tell me about my datasets`
- Run analysis:
  - Generate custom analysis: `analysis: [description]`
    (Generates Python code for analyzing selected dataset)
  - Execute code: `run datasetCode_[ID]` or `run.`
- Convert results: `convert dataset_[ID] to dataset`
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for dataset capabilities."""
        return """
Dataset Service Commands:
1. Information: 
   "tell me about my datasets"

2. Analysis Generation:
   "analysis: [description]"
   - Generates Python code for analyzing a selected dataset
   - Has access to all loaded datasets via 'datasets' dictionary
   - Uses pandas, numpy, plotly, scipy, sklearn
   - Supports visualization
   - Auto-validates code safety

3. Code Execution:
   "run datasetCode_[ID]" or "run." for last generated code
   - Executes specified code
   - May returns DataFrame results
   - May return a visualization
   - Provides previews and summaries

4. Result Conversion:
   "convert dataset_[ID] to dataset"
   - Saves analysis results as new dataset
   - Preserves metadata and execution history"""
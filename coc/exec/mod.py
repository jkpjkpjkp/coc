LOG_FILE = 'data/log/exec.log'

import io
import sys
import traceback
import textwrap
import ast
from langchain_core.tools import BaseTool
import copy
import os
import logging
from datetime import datetime
import pickle
import warnings
import contextlib

from pydantic import Field

class Exec:
    """Python code interpreter with proper initialization."""
    name: str = 'exec'
    description: str = (
        'Execute python code snippets with history contexts, '
        'supporting Jupyter-style last-line expression evaluation'
    )
    globals: dict

    def __init__(self, *args, task=None, **kwargs):
        """Initialize with essential globals and parent fields."""

        # Set up execution environment
        self.globals = {}
        self.globals.update({
            '__name__': '__main__',
            '__builtins__': __builtins__,
        })

        # Execute initialization code
        for code_path in args:
            with open(code_path, 'r') as f:
                self._run(f.read())

        # Add task to environment if provided
        if task:
            self.globals['task'] = copy.deepcopy(task)
            # Also set up individual image variables (image_1, image_2, etc.)
            if 'images' in task:
                for i, img in enumerate(task['images']):
                    self.globals[f'image_{i+1}'] = img



    def _run(self, code):
        """Execute code Jupyter-style: exec all but last statement; eval/display last expression.
        Accepts either a single code string or a list of code strings to execute.
        
        Args:
            code: A string containing Python code or a list of code strings to execute
            
        Returns:
            (stdout, stderr, images), ignoring FutureWarnings in stderr.
        """
        # Define a custom showwarning function to suppress FutureWarnings
        def custom_showwarning(message, category, filename, lineno, file=None, line=None):
            # Only print warnings to stderr if they are not FutureWarnings
            if not issubclass(category, FutureWarning):
                print(f"{filename}:{lineno}: {category.__name__}: {message}", file=file or sys.stderr)
        
        # Set up redirection for stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        # List to store image data (PNG bytes)
        displayed_images = []

        # Custom display function to handle display() calls
        def display(*args):
            for arg in args:
                if hasattr(arg, '_repr_png_'):
                    png_data = arg._repr_png_()
                    if png_data is not None:
                        displayed_images.append(png_data)
                else:
                    print(repr(arg))

        # Set the custom display function in globals
        self.globals['display'] = display

        # Handle list of code blocks by executing each block separately
        if isinstance(code, list):
            # Process each code block in the list
            for block in code:
                if not isinstance(block, str):
                    continue  # Skip non-string items
                
                # Prepare the code string
                block_str = '\n'.join(line.rstrip() for line in textwrap.dedent(str(block)).splitlines())
                
                # Execute the code block with stdout/stderr redirection
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), warnings.catch_warnings(record=True):
                    warnings.showwarning = custom_showwarning
                    try:
                        # Parse the code into an AST
                        body = ast.parse(block_str, mode='exec').body
                        if body:
                            # Separate initial statements from the last one
                            *initial, last = body
                            # Execute all statements except the last
                            if initial:
                                exec(compile(ast.Module(body=initial, type_ignores=[]), '<ast>', 'exec'), self.globals)
                            # Handle the last statement: evaluate if expression, execute otherwise
                            if isinstance(last, ast.Expr):
                                val = eval(compile(ast.Expression(last.value), '<ast>', 'eval'), self.globals)
                                if val is not None:
                                    if hasattr(val, '_repr_png_'):
                                        png_data = val._repr_png_()
                                        if png_data is not None:
                                            displayed_images.append(png_data)
                                    else:
                                        print(repr(val))
                                self.globals['_'] = val
                            else:
                                exec(compile(ast.Module(body=[last], type_ignores=[]), '<ast>', 'exec'), self.globals)
                    except Exception:
                        traceback.print_exc(file=stderr)
                        # Stop processing further blocks if there's a syntax error
                        if 'SyntaxError' in stderr.getvalue():
                            break
        else:
            # Single code string case - prepare the code string
            code_str = '\n'.join(line.rstrip() for line in textwrap.dedent(str(code)).splitlines())
            
            # Execute with stdout/stderr redirection
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), warnings.catch_warnings(record=True):
                warnings.showwarning = custom_showwarning
                try:
                    # Parse the code into an AST
                    body = ast.parse(code_str, mode='exec').body
                    if body:
                        # Separate initial statements from the last one
                        *initial, last = body
                        # Execute all statements except the last
                        if initial:
                            exec(compile(ast.Module(body=initial, type_ignores=[]), '<ast>', 'exec'), self.globals)
                        # Handle the last statement: evaluate if expression, execute otherwise
                        if isinstance(last, ast.Expr):
                            val = eval(compile(ast.Expression(last.value), '<ast>', 'eval'), self.globals)
                            if val is not None:
                                if hasattr(val, '_repr_png_'):
                                    png_data = val._repr_png_()
                                    if png_data is not None:
                                        displayed_images.append(png_data)
                                else:
                                    print(repr(val))
                            self.globals['_'] = val
                        else:
                            exec(compile(ast.Module(body=[last], type_ignores=[]), '<ast>', 'exec'), self.globals)
                except Exception:
                    traceback.print_exc(file=stderr)

        # Clean up the errors string
        errors = stderr.getvalue().strip()
        errors = '\n'.join(line for line in errors.split('\n') if line.strip())

        output = stdout.getvalue()
        if not output and errors:
            with open(LOG_FILE, 'a') as f:
                f.write(f'{datetime.now().strftime("%m/%d %H:%M:%S")}\n')
                f.write(str(code))
                f.write('\n')
                f.write('Errors:\n')
                f.write(errors)
                f.write('\n')

        return (output, errors, displayed_images)

    def get_var(self, name: str):
        return self.globals[name]

    def set_var(self, name: str, value):
        """Add a new variable into the execution environment."""
        self.globals[name] = value
    def __deepcopy__(self, memo):
        """Custom deep copy method that handles globals carefully.

        This method creates a new Exec instance and copies only
        serializable global variables, logging skipped items.
        """
        # Setup logging
        log_dir = os.path.join(os.getcwd(), 'data/log/deepcopy')
        os.makedirs(log_dir, exist_ok=True)

        # Create a unique log filename with timestamp
        log_filename = os.path.join(log_dir, f'deepcopy_skipped_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        # Configure logging
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Create a new instance of the class
        new_exec = self.__class__()

        # Copy the name and description
        new_exec.name = copy.deepcopy(self.name)
        new_exec.description = copy.deepcopy(self.description)

        # Create a new globals dictionary with only serializable items
        new_exec.globals = {}
        skipped_items = {}

        for key, value in self.globals.items():
            try:
                # Try to deep copy the value
                new_exec.globals[key] = copy.deepcopy(value)
            except (TypeError, pickle.PicklingError) as e:
                # Log detailed information about skipped items
                skipped_items[key] = {
                    'type': type(value).__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.warning(f"Skipped item: {key}")
                logger.warning(f"Type: {type(value).__name__}")
                logger.warning(f"Error: {e}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")

        # Log summary of skipped items
        if skipped_items:
            logger.info(f"Total skipped items: {len(skipped_items)}")
            logger.info("Skipped items details:")
            for key, details in skipped_items.items():
                logger.info(f"{key}: {details}")

        # Ensure '__name__' is set in the new globals
        new_exec.globals['__name__'] = '__main__'

        return new_exec
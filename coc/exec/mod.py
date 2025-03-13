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

from pydantic import Field

class Exec(BaseTool):
    """Python code interpreter with proper initialization."""
    name: str = 'exec'
    description: str = (
        'Execute python code snippets with history contexts, '
        'supporting Jupyter-style last-line expression evaluation'
    )
    globals: dict = Field(default_factory=dict)

    def __init__(self, *args, task=None, **kwargs):
        """Initialize with essential globals and parent fields."""
        # Initialize parent with required fields using class defaults
        super().__init__(**kwargs)

        # Set up execution environment
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

    def _run(self, code):
        """Execute code in a Jupyter-style way.

        - All statements but the last are executed normally (exec).
        - If the last statement is an expression, evaluate and print its result(s).
        - If the last expression is comma-separated, each sub-expression is printed on its own line.

        - returns (stdout, stderr)
        """

        code_str = textwrap.dedent(code)

        # remove trailing whitespace from each line
        code_str = '\n'.join(line.rstrip() for line in code_str.split('\n'))

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr

        errors = ''

        try:
            # Capture warnings during execution
            with warnings.catch_warnings(record=True) as captured_warnings:
                # Parse code into an AST
                parsed = ast.parse(code_str, mode='exec')
                body = parsed.body

                # Separate all but the last statement from the last statement
                *initial_statements, final_statement = body

                # 1) Execute all statements but the last
                if initial_statements:
                    initial_module = ast.Module(body=initial_statements, type_ignores=[])
                    compiled_init = compile(initial_module, filename='<ast>', mode='exec')
                    exec(compiled_init, self.globals)

                # 2) Jupyter-style evaluate/print for the last statement if it's an expression
                if isinstance(final_statement, ast.Expr):
                    last_expr = final_statement.value

                    # If it's a tuple like '1, 2, 3', evaluate each item
                    if isinstance(last_expr, ast.Tuple):
                        results = []
                        for element in last_expr.elts:
                            expr_code = ast.Expression(element)
                            compiled_expr = compile(expr_code, filename='<ast>', mode='eval')
                            val = eval(compiled_expr, self.globals)
                            results.append(val)
                        # Print each result on a new line
                        for r in results:
                            if r is not None:
                                print(r)
                        self.globals['_'] = tuple(results)
                    else:
                        # Single expression
                        expr_code = ast.Expression(last_expr)
                        compiled_expr = compile(expr_code, filename='<ast>', mode='eval')
                        val = eval(compiled_expr, self.globals)
                        if val is not None:
                            print(val)
                            self.globals['_'] = val
                else:
                    # If the last statement is not an expression, just exec it
                    final_module = ast.Module(body=[final_statement], type_ignores=[])
                    compiled_final = compile(final_module, filename='<ast>', mode='exec')
                    exec(compiled_final, self.globals)

        except Exception as e:
            errors += str(e)
            traceback.print_exc(file=redirected_stderr)

        output = redirected_stdout.getvalue()
        errors += redirected_stderr.getvalue()

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Process captured warnings: filter for FutureWarnings
        future_warnings = [fw for fw in captured_warnings if issubclass(fw.category, FutureWarning)]

        # Log FutureWarnings and/or errors if either exists
        if future_warnings or errors:
            with open(LOG_FILE, 'a') as f:
                from datetime import datetime
                f.write(f'{datetime.now().strftime("%m/%d %H:%M:%S")}\n')
                f.write(code)
                f.write('\n')
                if future_warnings:
                    f.write('FutureWarnings:\n')
                    for fw in future_warnings:
                        f.write(str(fw.message) + '\n')
                    f.write('\n')
                if errors:
                    f.write('Errors:\n')
                    f.write(errors)
                    f.write('\n')

        return (output, errors)

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
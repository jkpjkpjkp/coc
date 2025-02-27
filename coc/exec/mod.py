import io
import sys
import traceback
import textwrap
import ast
from langchain_core.tools import BaseTool
import logging
import copy

class Exec(BaseTool):
    name: str = 'exec'
    description: str = (
        'Execute python code snippets '
        'with history contexts, supporting Jupyter-style last-line expression evaluation'
    )
    globals: dict

    def __init__(self, *args):
        """Initialize a code environment.

        args: list of code files to execute in order.
        """
        super().__init__(globals={})
        for code_path in args:
            with open(code_path, 'r') as f:
                init_code = f.read()
            self._run(init_code)

    def _run(self, code):
        """Execute code in a Jupyter-style way.

           - All statements but the last are executed normally (exec).
           - If the last statement is an expression, evaluate and print its result(s).
           - If the last expression is comma-separated, each sub-expression is printed on its own line.

           - stdout and stderr are concatenated.
        """
        code_str = textwrap.dedent(code)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        try:
            sys.stdout = redirected_stdout
            sys.stderr = redirected_stderr

            try:
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
                    # final_statement.value is the actual expression AST node
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
                        # Optionally, store the entire tuple in `_`
                        self.globals['_'] = tuple(results)

                    else:
                        # Single expression
                        expr_code = ast.Expression(last_expr)
                        compiled_expr = compile(expr_code, filename='<ast>', mode='eval')
                        val = eval(compiled_expr, self.globals)
                        if val is not None:
                            print(val)
                            # Store it so we can retrieve it from self.globals if needed
                            self.globals['_'] = val

                else:
                    # If the last statement is not an expression, just exec it
                    final_module = ast.Module(body=[final_statement], type_ignores=[])
                    compiled_final = compile(final_module, filename='<ast>', mode='exec')
                    exec(compiled_final, self.globals)

            except Exception:
                # If anything inside code execution raises an exception
                traceback.print_exc(file=redirected_stderr)

            # Collect outputs
            output = redirected_stdout.getvalue()
            errors = redirected_stderr.getvalue()

        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if errors:
            logging.error(f'Errors: {errors}')

        return output + errors

    def add_var(self, name: str, value):
        """Add a new variable into the execution environment."""
        self.globals[name] = value


    def clone(self):
        """Create a deep copy of the current execution environment."""
        new_exec = Exec()
        new_exec.globals = copy.deepcopy(self.globals)
        return new_exec
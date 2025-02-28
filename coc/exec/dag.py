import ast
from collections import defaultdict
from .symbol_extraction import parse_cell_for_symbols
def build_dependency_dag(cells):
    """Build a DAG where edges represent symbol dependencies between cells."""
    dag = {i: set() for i in range(len(cells))}
    last_defined = defaultdict(int)  # Maps symbol to latest cell defining it

    for i, code in enumerate(cells):
        ret = parse_cell_for_symbols(code)
        defined = ret['defined']
        used = ret['used']
        # Collect dependencies for current cell
        dependencies = set()
        for symbol in used:
            if symbol in last_defined:
                dependencies.add(last_defined[symbol])
        # Ensure dependencies are cells before current
        valid_deps = {dep for dep in dependencies if dep < i}
        dag[i] = valid_deps
        # Update last_defined with current cell's symbols
        for symbol in defined:
            last_defined[symbol] = i
    return dag


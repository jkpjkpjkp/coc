import ast
from collections import defaultdict

class SymbolTracker(ast.NodeVisitor):
    def __init__(self):
        self.scope_stack = [set()]
        self.defined_symbols = set()
        self.used_symbols = set()

    def visit_FunctionDef(self, node):
        if len(self.scope_stack) == 1:
            self.defined_symbols.add(node.name)
        assigned_vars = self._collect_assigned_vars(node)
        assigned_vars.update(arg.arg for arg in node.args.args)
        self.scope_stack.append(assigned_vars)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ClassDef(self, node):
        if len(self.scope_stack) == 1:
            self.defined_symbols.add(node.name)
        assigned_vars = self._collect_assigned_vars(node)
        self.scope_stack.append(assigned_vars)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _collect_assigned_vars(self, node):
        assigned = set()
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        assigned.update(self._get_names(target))
                elif isinstance(child, (ast.For, ast.AsyncFor)):
                    assigned.update(self._get_names(child.target))
                elif isinstance(child, (ast.With, ast.AsyncWith)):
                    for item in child.items:
                        if item.optional_vars:
                            assigned.update(self._get_names(item.optional_vars))
                elif isinstance(child, ast.AugAssign):
                    assigned.update(self._get_names(child.target))
        return assigned

    def _get_names(self, node):
        names = set()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                names.update(self._get_names(elt))
        return names

    def visit_Assign(self, node):
        # Process the RHS and LHS before adding to defined_symbols
        self.generic_visit(node)
        if len(self.scope_stack) == 1:
            for target in node.targets:
                self._add_to_defined(target)

    def _add_to_defined(self, target):
        names = self._get_names(target)
        self.defined_symbols.update(names)

    def visit_Import(self, node):
        # Process child nodes first, though in reality Import has no children affecting this.
        self.generic_visit(node)
        if len(self.scope_stack) == 1:
            for alias in node.names:
                name = alias.asname or alias.name.split('.', 1)[0]
                self.defined_symbols.add(name)

    def visit_ImportFrom(self, node):
        # Process child nodes first, though similar to Import
        self.generic_visit(node)
        if len(self.scope_stack) == 1:
            for alias in node.names:
                name = alias.asname or alias.name
                self.defined_symbols.add(name)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            name = node.id
            in_scope = any(name in scope for scope in reversed(self.scope_stack))
            if not in_scope and name not in self.defined_symbols:
                self.used_symbols.add(name)

    # Handle list comprehensions and other comprehensions
    def visit_ListComp(self, node):
        self._handle_comp(node)

    def visit_SetComp(self, node):
        self._handle_comp(node)

    def visit_DictComp(self, node):
        self._handle_comp(node)

    def visit_GeneratorExp(self, node):
        self._handle_comp(node)

    def _handle_comp(self, node):
        assigned_vars = set()
        for gen in getattr(node, 'generators', []):
            assigned_vars.update(self._get_names(gen.target))
        self.scope_stack.append(assigned_vars)
        self.generic_visit(node)
        self.scope_stack.pop()

def parse_cell_for_symbols(code):
    tree = ast.parse(code)
    tracker = SymbolTracker()
    tracker.visit(tree)
    return {
        'defined': tracker.defined_symbols,
        'used': tracker.used_symbols
    }

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


import sys
import os

# Project root is the current working directory (set by VSCode task)
project_root = os.getcwd()
current_file = sys.argv[1]  # Absolute path of the current file from VSCode
rel_path = os.path.relpath(current_file, project_root)

# From source file to test file
if rel_path.startswith('coc/') and rel_path.endswith('.py'):
    test_rel_path = rel_path.replace('coc/', 'tests/', 1)
    test_rel_path = os.path.join(os.path.dirname(test_rel_path), 'test_' + os.path.basename(test_rel_path))
    test_path = os.path.join(project_root, test_rel_path)
    os.system(f'code -g {test_path}')

# From test file to source file
elif rel_path.startswith('tests/') and rel_path.endswith('.py') and os.path.basename(rel_path).startswith('test_'):
    source_rel_path = rel_path.replace('tests/', 'coc/', 1)
    source_filename = os.path.basename(source_rel_path)[5:]  # Remove 'test_'
    source_rel_path = os.path.join(os.path.dirname(source_rel_path), source_filename)
    source_path = os.path.join(project_root, source_rel_path)
    os.system(f'code -g {source_path}')

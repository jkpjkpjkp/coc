import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        # Get the project root directory (where the script is run)
        project_root = os.path.abspath('.')
        # Compute the relative path of the created file/directory from the project root
        rel_path = os.path.relpath(event.src_path, project_root)
        
        # Handle new directories
        if event.is_directory:
            init_file = os.path.join(event.src_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('')
        
        # Handle new Python files in coc/
        elif (rel_path.startswith('coc/') and 
              rel_path.endswith('.py') and 
              not rel_path.endswith('__init__.py')):
            # Construct the test file path
            test_rel_dir = os.path.dirname(rel_path).replace('coc/', 'tests/', 1)
            test_filename = 'test_' + os.path.basename(rel_path)
            test_path = os.path.join(project_root, test_rel_dir, test_filename)
            
            # Create the test file if it doesn’t exist
            if not os.path.exists(test_path):
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                # Compute the module path for the import statement
                module_path = rel_path[4:-3].replace('/', '.')  # Remove 'coc/' and '.py'
                with open(test_path, 'w') as f:
                    f.write(f'import pytest\n')
                    f.write(f'import coc.{module_path}\n\n')
                    f.write(f'def test_placeholder():\n')
                    f.write(f'    pass\n')

if __name__ == "__main__":
    path = '.'  # Monitor the project root
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        # Handle new directories
        if event.is_directory:
            init_file = os.path.join(event.src_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('')
        # Handle new Python files in coc/
        elif (event.src_path.startswith('coc/') and 
              event.src_path.endswith('.py') and 
              not event.src_path.endswith('__init__.py')):
            rel_path = os.path.relpath(event.src_path, 'coc/')
            test_dir = os.path.join('tests', os.path.dirname(rel_path))
            test_filename = 'test_' + os.path.basename(rel_path)
            test_path = os.path.join(test_dir, test_filename)
            if not os.path.exists(test_path):
                os.makedirs(test_dir, exist_ok=True)
                module_path = '.'.join(os.path.dirname(rel_path).split(os.sep) + 
                                     [os.path.splitext(os.path.basename(rel_path))[0]])
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

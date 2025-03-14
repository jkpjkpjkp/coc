import threading

# Configuration
MAX_PARALLEL = 4  # Maximum number of concurrent threads (adjust based on your needs)
_semaphore = threading.Semaphore(MAX_PARALLEL)
_local = threading.local()

class ConcurrentTool:
    def __init__(self, tool_factory):
        """Wrapped tool with thread-safe lazy loading and concurrency control.

        Args:
            tool_factory: A callable that returns the tool instance when called.
        """
        self._tool_factory = tool_factory
        self._tool = None
        self._init_lock = threading.Lock()

    def _initialize_tool(self):
        """Initialize the tool in a thread-safe manner. """
        with self._init_lock:
            # Double-check locking: ensure another thread hasn’t initialized it while waiting
            if self._tool is None:
                self._tool = self._tool_factory()

    def __getattr__(self, name):
        """Handle attribute access, initializing the tool if necessary and wrapping methods for concurrency.

        Args:
            name: The name of the attribute or method being accessed.
        """
        # Ensure the tool is initialized
        if self._tool is None:
            self._initialize_tool()

        # Get the attribute from the tool
        attr = getattr(self._tool, name)

        # If it’s a callable (method), wrap it with concurrency control
        if callable(attr):
            def wrapped(*args, **kwargs):
                # Initialize thread-local semaphore count if not set
                try:
                    count = _local.semaphore_count
                except AttributeError:
                    _local.semaphore_count = 0
                    count = 0

                # Acquire semaphore only for the top-level call in this thread
                if count == 0:
                    _semaphore.acquire()
                _local.semaphore_count += 1

                try:
                    # Call the original method
                    return attr(*args, **kwargs)
                finally:
                    # Clean up: decrement count and release semaphore when done
                    _local.semaphore_count -= 1
                    if _local.semaphore_count == 0:
                        _semaphore.release()
            return wrapped
        else:
            # Return non-callable attributes as-is
            return attr
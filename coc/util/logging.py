import sys
import os
from datetime import datetime
from pathlib import Path
from loguru import logger

# Create log directory structure
LOG_DIR = Path("data/log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default handler
logger.remove()

# Add a handler to stderr with a specific format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Configure default log file based on current timestamp
DEFAULT_LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(
    DEFAULT_LOG_FILE,
    rotation="10 MB",
    retention="1 week",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO"
)

def setup_logger(name, log_file=None):
    """
    Set up a logger with specific configuration.
    If log_file is provided, adds a file handler to that specific file.
    Returns the configured logger.
    """
    # If a specific log file is provided, add a file handler
    if log_file:
        log_path = LOG_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            filter=lambda record: record["extra"].get("name") == name
        )
    
    # Return a contextualized logger
    return logger.bind(name=name)

# Compatibility layer with standard logging module
def get_logger(name, log_file=None):
    """
    Compatibility function that mimics logging.getLogger but returns a loguru logger.
    """
    return setup_logger(name, log_file)

# Import logging here to avoid circular imports
import logging

# Handler class that properly inherits from logging.Handler
class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging calls and redirects them to loguru.
    This handler preserves compatibility with the standard logging module.
    """
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        
    def emit(self, record):
        # Skip log records emitted by loguru to avoid infinite recursion
        if record.name == "loguru":
            return
            
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
            
        # Find caller from where this record was issued
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_intercept():
    """
    Sets up interception of standard logging to redirect to loguru
    without replacing the original logging module functionality.
    This preserves compatibility with libraries that use logging directly.
    """
    # Remove existing InterceptHandler if any
    logging.root.handlers = [h for h in logging.root.handlers if not isinstance(h, InterceptHandler)]
    
    # Create the intercepting handler
    handler = InterceptHandler()
    
    # Add it to the root logger
    logging.root.addHandler(handler)
    
    # Set the root logger level to ensure all messages are intercepted
    logging.root.setLevel(logging.INFO)

# Set up the interception by default
setup_intercept() 
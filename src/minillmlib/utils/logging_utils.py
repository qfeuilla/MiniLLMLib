"""Logging utilities for MiniLLMLib."""
import logging
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None

def get_logger(name: str = "minillmlib") -> logging.Logger:
    """Get or create a logger instance.
    
    If a logger with the given name already exists, returns that logger without
    modifying its configuration. If no logger exists, creates a new one with
    basic configuration.
    
    Args:
        name: The name of the logger. Defaults to "minillmlib".
        
    Returns:
        A Logger instance.
    """
    global _logger # pylint: disable=global-statement

    if _logger is not None:
        return _logger

    # Get or create logger
    logger = logging.getLogger(name)

    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

        logging.getLogger("httpx").setLevel(logging.WARNING)

    _logger = logger
    return logger

def configure_logger(
    level: Optional[int] = None,
    format_class: Optional[logging.Formatter] = None,
    handlers: Optional[list[logging.Handler]] = None
) -> None:
    """Configure the global logger instance.
    
    Args:
        level: The logging level to set. If None, keeps current level.
        format_class: A custom formatter class. If None, keeps current formatter.
        handlers: List of handlers to use. If None, keeps current handlers.
    """
    logger = get_logger()

    if level is not None:
        logger.setLevel(level)

    if handlers is not None:
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add new handlers
        for handler in handlers:
            if format_class is not None:
                handler.setFormatter(format_class())
            logger.addHandler(handler)
    elif format_class is not None:
        # Just update formatter on existing handlers
        for handler in logger.handlers:
            handler.setFormatter(format_class())

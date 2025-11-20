"""Structured logging module for GenAI CAD/CFD Studio.

This module provides a comprehensive logging setup with colored console output,
log rotation, and context-aware logging for better tracing and debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
import contextvars


# Context variable for storing trace context
trace_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'trace_context', default={}
)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output.

    Different log levels are displayed in different colors for better readability.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format.

        Returns:
            Formatted and colored log message.
        """
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # Add trace context if available
        context = trace_context.get()
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            record.msg = f"[{context_str}] {record.msg}"

        return super().format(record)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that includes context information in log messages.

    This adapter automatically adds context information (like request ID,
    user ID, etc.) to all log messages.
    """

    def process(self, msg: str, kwargs: Any) -> tuple:
        """Process log message and add context.

        Args:
            msg: Log message.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of (message, kwargs) with context added.
        """
        context = trace_context.get()
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            msg = f"[{context_str}] {msg}"

        return msg, kwargs


def setup_logger(
    name: str = "genai_cad_cfd",
    log_level: str = "INFO",
    log_dir: Optional[str] = "logs",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up and configure a logger with console and file handlers.

    Args:
        name: Logger name. Defaults to "genai_cad_cfd".
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_dir: Directory for log files. Defaults to "logs".
        console_output: Enable colored console output. Defaults to True.
        file_output: Enable file output with rotation. Defaults to True.
        max_bytes: Maximum size of log file before rotation (bytes).
            Defaults to 10MB.
        backup_count: Number of backup log files to keep. Defaults to 5.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger("my_app", log_level="DEBUG")
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))

        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def set_trace_context(**kwargs: Any) -> None:
    """Set trace context for current execution context.

    This context will be automatically included in all log messages
    within the current async context or thread.

    Args:
        **kwargs: Key-value pairs to include in trace context.

    Example:
        >>> set_trace_context(request_id="abc123", user_id="user456")
        >>> logger.info("Processing request")  # Will include context
    """
    current_context = trace_context.get().copy()
    current_context.update(kwargs)
    trace_context.set(current_context)


def clear_trace_context() -> None:
    """Clear the trace context for current execution context."""
    trace_context.set({})


def get_trace_context() -> Dict[str, Any]:
    """Get the current trace context.

    Returns:
        Dictionary containing current trace context.
    """
    return trace_context.get().copy()


class LoggerContext:
    """Context manager for temporary logging context.

    Example:
        >>> with LoggerContext(request_id="abc123"):
        ...     logger.info("Processing")  # Will include request_id
        >>> logger.info("Done")  # Won't include request_id
    """

    def __init__(self, **kwargs: Any):
        """Initialize context manager.

        Args:
            **kwargs: Context key-value pairs.
        """
        self.context = kwargs
        self.previous_context: Dict[str, Any] = {}

    def __enter__(self) -> 'LoggerContext':
        """Enter context and set trace context."""
        self.previous_context = get_trace_context()
        set_trace_context(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore previous trace context."""
        trace_context.set(self.previous_context)


def get_logger(
    name: str,
    log_level: Optional[str] = None,
    **kwargs
) -> logging.Logger:
    """Get or create a logger with the specified configuration.

    Args:
        name: Logger name.
        log_level: Logging level. If None, uses INFO.
        **kwargs: Additional arguments passed to setup_logger.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    logger = logging.getLogger(name)

    # Only setup if logger has no handlers (avoid duplicate setup)
    if not logger.handlers:
        return setup_logger(name, log_level or "INFO", **kwargs)

    return logger


# Create default application logger
default_logger = setup_logger()


# Convenience functions using default logger
def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message using default logger."""
    default_logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message using default logger."""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message using default logger."""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message using default logger."""
    default_logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message using default logger."""
    default_logger.critical(msg, *args, **kwargs)

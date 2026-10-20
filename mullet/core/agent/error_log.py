"""Configure rotating file-based error logging for the Mullet agent."""

import logging
import os
from logging.handlers import RotatingFileHandler

def get_error_log_path():
    """
    Returns the absolute path to the error log file.
    """
    home_dir = os.path.expanduser("~")
    log_dir = os.path.join(home_dir, ".mullet", "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "errors.log")

def setup_error_log():
    """
    Configures the root logger to use a RotatingFileHandler for error logging.
    Logs messages with WARNING severity or higher to a persistent file,
    with a max size of 5MB and 3 backup files.
    """
    log_file = get_error_log_path()
    
    # Create a rotating file handler
    # Max size 5MB, keep 3 backup files
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3
    )
    
    # Set the logging level for this handler
    file_handler.setLevel(logging.WARNING)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Add the handler to the root logger only if it's not already present
    # This prevents duplicate handlers if setup_error_log is called multiple times
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == log_file for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
    
    # Ensure the root logger's level is at least WARNING,
    # so messages are not filtered out before reaching the handler
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.WARNING)

# Example usage (optional, for testing)
if __name__ == "__main__":  # pragma: no cover
    setup_error_log()
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.info("This info message should not be logged to the file.")
    print(f"Error log path: {get_error_log_path()}")
    print("Check the log file for WARNING and ERROR messages.")

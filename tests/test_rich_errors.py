from mullet.cli.rich_errors import format_error
from colorama import Fore, Style

def test_format_error_import_error():
    try:
        import non_existent_module_123  # noqa: F401  (intentional: test triggers ImportError)
    except ImportError as e:
        formatted_msg = format_error(e)
        assert f"{Fore.RED}Error: ModuleNotFoundError{Style.RESET_ALL}" in formatted_msg
        assert f"{Fore.YELLOW}Suggestion: Try running `pip install non_existent_module_123`{Style.RESET_ALL}" in formatted_msg

def test_format_error_file_not_found_error():
    try:
        with open("non_existent_file_123.txt", "r") as f:
            f.read()
    except FileNotFoundError as e:
        formatted_msg = format_error(e)
        assert f"{Fore.RED}Error: FileNotFoundError{Style.RESET_ALL}" in formatted_msg
        assert f"{Fore.YELLOW}Suggestion: Please check if the file path is correct and the file exists.{Style.RESET_ALL}" in formatted_msg

def test_format_error_connection_error():
    error = ConnectionError("Failed to connect")
    formatted_msg = format_error(error)
    assert f"{Fore.RED}Error: ConnectionError{Style.RESET_ALL}" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion: Check your network connection or the target server's status.{Style.RESET_ALL}" in formatted_msg

def test_format_error_value_error():
    error = ValueError("Invalid input")
    formatted_msg = format_error(error)
    assert f"{Fore.RED}Error: ValueError{Style.RESET_ALL}" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion: Review the input values. This error often indicates incorrect arguments.{Style.RESET_ALL}" in formatted_msg

def test_format_error_type_error():
    error = TypeError("Unsupported operand type")
    formatted_msg = format_error(error)
    assert f"{Fore.RED}Error: TypeError{Style.RESET_ALL}" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion: Check the types of arguments passed to functions or methods.{Style.RESET_ALL}" in formatted_msg

def test_format_error_generic_exception():
    error = Exception("Something unexpected happened")
    formatted_msg = format_error(error)
    assert f"{Fore.RED}Error: Exception{Style.RESET_ALL}" in formatted_msg
    assert "Something unexpected happened" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion:" not in formatted_msg # No specific suggestion for generic exception

def test_format_error_with_context():
    error = ValueError("Invalid input")
    context = "During processing user data"
    formatted_msg = format_error(error, context=context)
    assert f"{Fore.RED}Error: ValueError{Style.RESET_ALL}" in formatted_msg
    assert f"{Fore.CYAN}Context:{Style.RESET_ALL} {context}" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion: Review the input values. This error often indicates incorrect arguments.{Style.RESET_ALL}" in formatted_msg

def test_format_error_empty_error_message():
    error = ValueError("")
    formatted_msg = format_error(error)
    assert f"{Fore.RED}Error: ValueError{Style.RESET_ALL}" in formatted_msg
    assert f"{Fore.YELLOW}Suggestion: Review the input values. This error often indicates incorrect arguments.{Style.RESET_ALL}" in formatted_msg

import logfire

def parse_text(file_path: str):
    """
    Parses text content and extracts text from it.

    Returns:
        str: The extracted text from the text content.
    """
    with logfire.span("Parsing text content", file_path=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                return text
        except Exception as e:
            logfire.error("Error parsing text content", error=str(e), file_path=file_path)
            raise e
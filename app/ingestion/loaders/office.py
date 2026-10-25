import os
import logfire
from unstructured.partition.auto import partition 

def parse_office(file_path: str):
    """
    Parses Office document content and extracts text from it.

    Returns:
        str: The extracted text from the Office document.
    """
    with logfire.span("Parsing Office document content", file_path=file_path):
        try:
            elements = partition(file_path)
            full_text = "\n".join([element.text for element in elements if hasattr(element, "text")])

            if not full_text:
                logfire.warning("No text extracted from Office document", {file_path})
            else:
                logfire.info("Successfully extracted text from Office document", {len(full_text)})

            return full_text
        except Exception as e:
            logfire.error("Error parsing Office document content", error=str(e))
            raise e
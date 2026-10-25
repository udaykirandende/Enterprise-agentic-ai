import os
import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str):
    """
    Parses PDF content and extracts text from it.

    Returns:
        str: The extracted text from the PDF content.
    """
    with logfire.span("Parsing PDF content", file_path=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages in PDF {total_pages}")

            text_parts: list[str] = []
            blank_pages: list[int] = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)

            if blank_pages:
                logfire.info(f"Blank pages found in PDF: {blank_pages}")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text()
                            if fallback_text:
                                text_parts.append(fallback_text)
                                logfire.info(
                                    f"Text extracted from blank page {page_num} using pdfplumber"
                                )
                except Exception as plumber_error:
                    logfire.error(
                        f"Error using pdfplumber for blank pages: {str(plumber_error)}"
                    )

            return " ".join(text_parts)
        except Exception as error:
            logfire.error(f"Error parsing PDF content: {str(error)}")
            return ""


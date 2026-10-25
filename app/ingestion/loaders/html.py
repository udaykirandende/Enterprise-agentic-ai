from bs4 import BeautifulSoup
import logfire 

def parse_html(file_path: str) :
    """
    Parses HTML content and extracts text from it.

    Returns:
        str: The extracted text from the HTML content.
    """
    with logfire.span("Parsing HTML content",file_path=file_path):

        try:
           with open(file_path, "r", encoding="utf-8",errors="ignore") as f:
                content = f.read()
                soup = BeautifulSoup(content, "html.parser")

                # Remove script and style elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                # Get text
                text = soup.get_text(separator="\n")
                # Break into lines and remove leading/trailing space on each
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each    
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Drop blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)

                return text
        except Exception as e:
            logfire.error("Error parsing HTML content", error=str(e), file_path=file_path)
            return ""

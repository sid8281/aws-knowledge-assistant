"""Extract readable text from an HTML file using BeautifulSoup."""

from bs4 import BeautifulSoup


def load_html(file_path: str) -> str:
    """Strip tags/scripts/styles and return visible text from an HTML file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove elements that don't contain readable content
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

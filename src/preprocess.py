import re
from bs4 import BeautifulSoup
import contractions

''' --- Binary Label Encoding --- '''

label_map = {"positive": 1, "negative": 0}

""" --- Text Cleaning Pipeline --- """

def clean_text(text: str) -> str:

    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Expand contractions
    text = contractions.fix(text)

    # Lowercase
    text = text.lower()

    # Scrub digits
    text = re.sub(r"\d+", " ", text)

    # Strip special characters and symbols
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

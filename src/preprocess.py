import re

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_labels(series):
    # CG: Computer Generated (Fake) = 1, OR: Original (Real) = 0
    mapping = {'CG': 1, 'OR': 0}
    return series.map(mapping)
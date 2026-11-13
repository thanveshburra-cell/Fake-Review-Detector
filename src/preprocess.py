import re

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_labels(labels):
    mapping = {'CG': 1, 'OR': 0}
    return [mapping.get(str(lbl).strip(), None) for lbl in labels]
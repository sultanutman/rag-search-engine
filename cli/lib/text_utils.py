import string

from nltk.stem import PorterStemmer

from .search_utils import (
    load_stopwords,
)

def _preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def text_tokenization(text: str) -> list[str]:
    preprocessed_text = _preprocess_text(text)
    tokens = preprocessed_text.strip().split()
    valid_tokens = [token for token in tokens if token]
    filtered_tokens = _remove_stopwords(valid_tokens)
    return filtered_tokens

def _has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    return any(q in title for q in query_tokens for title in title_tokens)

STOPWORDS = [_preprocess_text(word) for word in load_stopwords()]

def _remove_stopwords(tokens: list[str]):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in STOPWORDS]

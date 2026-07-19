import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        title = movie['title']
        query_tokens = _text_tokenization(query)
        title_tokens = _text_tokenization(movie['title'])
        if _has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def _preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def _text_tokenization(text: str) -> list[str]:
    preprocessed_text = _preprocess_text(text)
    tokens = [token for token in preprocessed_text.strip().split(" ") if token.strip()]
    return tokens

def _has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    return any(q in title for q in query_tokens for title in title_tokens)
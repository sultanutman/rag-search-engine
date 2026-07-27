import os
import string
import pickle
from collections import defaultdict

from .search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    load_movies, 
    load_stopwords,
    CACHE_DIR, 
    INDEX_PATH, 
    DOC_MAP_PATH,
)
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text) -> None:
        tokens = text_tokenization(text)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term) -> list[int]:
        indexes = self.index.get(term, set())
        return sorted(indexes)       
    
    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            doc_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_text)
            self.docmap[doc_id] = movie
        
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOC_MAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        try: 
            with open(INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            with open(DOC_MAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            raise Exception('file not found')
        

def search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    local_inverted_index = InvertedIndex()
    local_inverted_index.load()
    
    seen = set()
    result = []
    for token in text_tokenization(query):
        doc_ids = local_inverted_index.get_documents(token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = local_inverted_index.docmap[doc_id]
            result.append(doc)
            if len(result) >= limit:
                return result[:limit]

def build_command():
    local_inverted_index = InvertedIndex()
    local_inverted_index.build()
    local_inverted_index.save()

def _preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def text_tokenization(text: str) -> list[str]:
    preprocessed_text = _preprocess_text(text)
    tokens = [token for token in preprocessed_text.strip().split(" ") if token.strip()]
    filtered_tokens = _remove_stopwords(tokens)
    return filtered_tokens

def _has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    return any(q in title for q in query_tokens for title in title_tokens)

STOPWORDS = [_preprocess_text(word) for word in load_stopwords()]

def _remove_stopwords(tokens: list[str]):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in STOPWORDS]

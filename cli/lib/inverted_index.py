import os
import pickle
import math
from collections import defaultdict, Counter

from .search_utils import (
    load_movies, 
    CACHE_DIR, 
    INDEX_PATH, 
    DOC_MAP_PATH,
    TERM_FREQ_PATH,
    DOC_LENGTHS_PATH,
    BM25_K1,
    BM25_B,
    Movie,
    SearchResult,
    format_search_results,
)
from .text_utils import (
    text_tokenization
)

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

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
        with open(TERM_FREQ_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        try: 
            with open(INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            with open(DOC_MAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
            with open(TERM_FREQ_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(DOC_LENGTHS_PATH, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            raise Exception('file not found')

    def get_documents(self, term) -> list[int]:
        indexes = self.index.get(term, set())
        return sorted(list(indexes))

    def __add_document(self, doc_id, text) -> None:
        tokens = text_tokenization(text)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)

        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def _tokenize_term(self, term) -> str:
        token = text_tokenization(term)
        if len(token) != 1:
            raise Exception('term must be one token size')
        return token[0]

    def get_tf(self, doc_id, term) -> int:
        token = self._tokenize_term(term)
        return self.term_frequencies[doc_id][token]

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        else:
            length_norm = 1
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def get_idf(self, term: str) -> float:
        token = self._tokenize_term(term)
        doc_count = len(self.docmap)
        term_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = self._tokenize_term(term)
        doc_count = len(self.docmap)
        term_count = len(self.index[token])
        return math.log((doc_count - term_count + 0.5) / (term_count +0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_lengths = 0
        for length in self.doc_lengths.values():
            total_lengths += length
        return total_lengths / len(self.doc_lengths)

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query, limit):
        query_tokens = text_tokenization(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_results(
                doc_id=doc["id"], 
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def get_movie_name(self, doc_id: int) -> str:
        movie = self.docmap.get(doc_id, None)
        if movie:
            return movie['title']
        return ""

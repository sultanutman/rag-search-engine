import os

from pickle import dump

from .keyword_search import text_tokenization
from .search_utils import load_movies, CACHE_DIR, INDEX_DIR, DOC_MAP_DIR

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, str]

    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = text_tokenization(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)


    def get_documents(self, term):
        indexes = self.index[term]
        if indexes:
            return sorted(indexes)
        return None
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'], doc_text)
            self.docmap[movie['id']] = doc_text
        
        self.save()
        
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_DIR, "wb") as f:
            dump(self.index, f)
        with open(DOC_MAP_DIR, "wb") as f:
            dump(self.docmap, f)
    


    
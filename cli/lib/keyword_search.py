from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B
)
from .text_utils import (
    text_tokenization,
)
from .inverted_index import InvertedIndex

def search_command(
    lii: InvertedIndex,
    query: str, 
    limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    lii.load()
    
    seen = set()
    result = []
    for token in text_tokenization(query):
        doc_ids = lii.get_documents(token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = lii.docmap[doc_id]
            result.append(doc)
            if len(result) >= limit:
                return result[:limit]

def build_command(
    lii: InvertedIndex,
):
    lii.build()
    lii.save()

def tf_search(
    lii: InvertedIndex,
    doc_id: int,
    term: str,
) -> int:
    lii.load()

    return lii.get_tf(doc_id, term)


def idf_search(
    lii: InvertedIndex,
    term: str,
) -> float:
    lii.load()

    return lii.get_idf(term)

def tf_idf_search(
    lii: InvertedIndex,
    doc_id: int,
    term: str,
) -> float:
    lii.load()

    tf = lii.get_tf(doc_id, term)
    idf = lii.get_idf(term)

    return tf * idf

def bm25_idf_command(
    lii: InvertedIndex,
    doc_id: int,
    term: str,
) -> float:
    lii.load()

    idf = lii.get_bm25_idf(term)

    return idf

def bm25_tf_command(
    lii: InvertedIndex,
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    lii.load()

    return lii.get_bm25_tf(doc_id, term, k1, b)

def bm_search_command(
    lii: InvertedIndex,
    query: str,
    limit: int = 5,
) -> list[dict[int, float]]:
    lii.load()
    return lii.bm25_search(query, limit)
    
    

    

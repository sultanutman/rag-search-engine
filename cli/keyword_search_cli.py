import argparse

from lib.keyword_search import (
    search_command, 
    build_command, 
    tf_search, 
    idf_search, 
    tf_idf_search,
    bm25_idf_command,
    bm25_tf_command,
    bm_search_command,
    )
from lib.inverted_index import InvertedIndex
from lib.search_utils import (
    BM25_K1, 
    BM25_B,
    )

def main() -> None:

    local_inverted_index = InvertedIndex()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build movies data search indexes")

    tf_parser = subparsers.add_parser("tf", help="Find frequency of term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Source document id to search")
    tf_parser.add_argument("term", type=str, help="Term you want to search frequency in doc_id")

    idf_parser = subparsers.add_parser("idf", help="Find inverse document frequency of term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF score for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Find inverse document frequency of term")
    tfidf_parser.add_argument("doc_id", type=int, help="Source document id to search")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Tunable BM25 search result limit"
    )

    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            results = search_command(local_inverted_index, args.query)
            for index, r in enumerate(results, 1):
                print(f'{index}. {r['title']}')
        case "build":
            print("Building inverted index...")
            build_command(local_inverted_index)
            print("Inverted index built successfully.")
        case "tf":
            frequency = tf_search(local_inverted_index, args.doc_id, args.term)
            print(f'Term: {args.term} found {frequency} times in the doc {args.doc_id}')
        case "idf":
            idf = idf_search(local_inverted_index, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_search(local_inverted_index, args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(local_inverted_index, 0, term=args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(local_inverted_index, args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            results = bm_search_command(local_inverted_index, args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
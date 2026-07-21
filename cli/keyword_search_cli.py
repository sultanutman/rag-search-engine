import argparse

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build movies data search indexes")

    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f'Searching for: {args.query}')
            results = search_command(args.query)
            for index, r in enumerate(results, 1):
                print(f'{index}. {r['title']}')
        case "build":
            local_inverted_index = InvertedIndex()
            local_inverted_index.build()
            docs = local_inverted_index.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
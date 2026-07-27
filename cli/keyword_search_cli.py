import argparse

from lib.keyword_search import search_command, build_command

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
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
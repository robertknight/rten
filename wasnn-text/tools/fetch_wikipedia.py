from argparse import ArgumentParser

import wikipediaapi as wiki


def main():
    parser = ArgumentParser(description="Fetch text of Wikipedia pages")
    parser.add_argument("page_name", help="Name of page to fetch")
    parser.add_argument("-o", "--output", help="Output filename")
    args = parser.parse_args()

    page_name = args.page_name.strip().replace(" ", "_")
    output_file = args.output or f"{page_name}.txt"

    wiki_wiki = wiki.Wikipedia("wasnn-text (robertknight@gmail.com)", "en")
    page = wiki_wiki.page(page_name)

    with open(output_file, "w") as output:
        output.write(page.text)


if __name__ == "__main__":
    main()

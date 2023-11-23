import sqlite3
import sys


def main(dbfile, doi, outfile):
    db = sqlite3.connect(dbfile)
    cursor = db.cursor()

    cursor.execute("select pdf_content from pdfs where doi = ?", (doi,))

    blob = cursor.fetchone()[0]

    with open(outfile, "wb") as fp:
        fp.write(blob)


if __name__ == "__main__":
    main(*sys.argv[1:])

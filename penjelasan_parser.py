#!/usr/bin/env python3
import re
import csv
from pdfminer.high_level import extract_text
from pathlib import Path

INPUT_FILES = [
    ("UU 11/2008", Path("UU_NO_11_2008-1_removed.pdf")),
    ("PP 71/2019", Path("PP_NO_71_2019_removed.pdf")),
    ("UU 1/2024", Path("UU_NO_1_2024_removed.pdf")),
]
OUTPUT_CSV = "penjelasan_master.csv"

# regex
RE_PASAL = re.compile(r'^Pasal\s+([0-9A-Za-z]+)$', re.IGNORECASE)
RE_AYAT  = re.compile(r'^Ayat\s*\(\s*([0-9A-Za-z]+)\s*\)$', re.IGNORECASE)
RE_PASAL_DEMI_PASAL = re.compile(r'^\s*(?:II\.\s*)?PASAL\s+DEMI\s+PASAL\s*$', re.IGNORECASE)

# untuk"Cukup jelas."
RE_STANDALONE_CUKUP = re.compile(r'^\s*cukup jelas\.?\s*$', re.IGNORECASE)

# huruf/angka declarations
RE_HURUF_ONLY = re.compile(r'^Huruf\s+[A-Za-z0-9]+\s*$', re.IGNORECASE)
RE_ANGKA_ONLY = re.compile(r'^Angka\s+[A-Za-z0-9]+\s*$', re.IGNORECASE)

# headers footers
RE_PAGE_FOOTER = re.compile(r'^\s*\d+\s*/\s*\d+\s*$')
HUKUMONLINE_HEADERS = [
    "www.hukumonline.com",
    "www.hukumonline.com/pusatdata"
]

def extract_lines_from_pdf(path: Path):
    raw = extract_text(str(path))
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

def is_header_or_footer(line: str):
    # page footer
    if RE_PAGE_FOOTER.match(line):
        return True
    # hukumonline headers
    l = line.lower().strip()
    return l in ("www.hukumonline.com", "www.hukumonline.com/pusatdata")

def is_pasaldemic(line: str):
    return bool(RE_PASAL_DEMI_PASAL.match(line))

def match_pasal(line: str):
    m = RE_PASAL.fullmatch(line)
    return m.group(1) if m else None

def match_ayat(line: str):
    m = RE_AYAT.fullmatch(line)
    return m.group(1) if m else None

def is_cukup(line: str):
    return bool(RE_STANDALONE_CUKUP.match(line))

def is_empty_huruf_or_angka(line: str):
    return bool(RE_HURUF_ONLY.match(line) or RE_ANGKA_ONLY.match(line))

# main
def parse_penjelasan(source_name: str, pdf_path: Path, writer):

    lines = extract_lines_from_pdf(pdf_path)
    if not lines:
        print(f"[WARN] No text in {pdf_path}")
        return

    # remove headers/footers FIRST
    clean_lines = []
    for ln in lines:
        if is_header_or_footer(ln):
            continue
        clean_lines.append(ln)
    lines = clean_lines

    # find PASAL DEMI PASAL start
    start_idx = None
    for i, ln in enumerate(lines):
        if is_pasaldemic(ln):
            start_idx = i
            break
    if start_idx is None:
        print(f"[WARN] PASAL DEMI PASAL not found in {pdf_path}")
        return

    i = start_idx + 1
    current_pasal = None
    current_ayat = None
    buffer = []

    def flush():
        nonlocal buffer, current_pasal, current_ayat
        if current_pasal is None:
            buffer = []
            return
        ay = current_ayat if current_ayat else "1"
        merged = " ".join(buffer).strip()
        if not merged:
            buffer = []
            return
        if is_cukup(merged):
            buffer = []
            return
        writer.writerow([source_name, current_pasal, ay, merged])
        buffer = []

    # Iterate lines
    while i < len(lines):
        ln = lines[i]

        # skip empty Huruf/Angka lines (new rule)
        if is_empty_huruf_or_angka(ln):
            # only skip if next line is NOT real explanation
            next_ln = lines[i + 1] if i + 1 < len(lines) else ""
            if (
                match_pasal(next_ln) or
                match_ayat(next_ln) or
                is_empty_huruf_or_angka(next_ln) or
                is_cukup(next_ln) or
                next_ln == ""
            ):
                i += 1
                continue

        # detect Pasal
        p = match_pasal(ln)
        if p:
            flush()
            current_pasal = p
            current_ayat = None
            buffer = []
            i += 1
            continue

        # detect Ayat
        a = match_ayat(ln)
        if a:
            flush()
            current_ayat = a
            buffer = []
            i += 1
            continue

        # "Cukup jelas."
        if is_cukup(ln):
            i += 1
            continue

        # normal text
        if current_pasal:
            buffer.append(ln)

        i += 1

    flush()

# start here
def main():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "pasal", "ayat", "text"])
        for src, pdf_path in INPUT_FILES:
            parse_penjelasan(src, pdf_path, writer)

    print("Done →", OUTPUT_CSV)

if __name__ == "__main__":
    main()


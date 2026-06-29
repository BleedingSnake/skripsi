#!/usr/bin/env python3
"""
pasal_master_simple.py

Simple, top-down parser for:
 - UU 11/2008  -> UU_NO_11_2008-1.PDF
 - PP 71/2019  -> PP_NO_71_2019.PDF
 - UU 1/2024   -> UU_NO_1_2024.pdf

Implements the exact rules you provided (Option 1, patched):
 - start markers, stop markers
 - header/footer removal
 - implicit ayat -> set ayat = 1 when no numeric bullets in a pasal
 - combine duplicate rows by concatenating text (document order)
"""

import re
import pdfplumber
import pandas as pd
import os

# utils 
def extract_lines_from_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_lines = text.splitlines()
            for ln in page_lines:
                s = ln.strip()
                if not s:
                    continue
                # remove header containing hukumonline link
                if re.search(r'hukumonline', s, re.IGNORECASE):
                    continue
                # remove standalone page numbers
                if re.fullmatch(r"\d{1,4}", s):
                    continue
                # remove common 'typo report' lines
                if re.search(r"Menemukan kesalahan ketik", s, re.IGNORECASE):
                    continue
                lines.append(s)
            # add explicit page break marker (helps debugging / keeps order)
            lines.append("<PAGE_BREAK>")
    return lines

# detection helpers 
RE_BAB = re.compile(r'^BAB\s+([IVXLCDM]+)$', re.IGNORECASE)
RE_PASAL = re.compile(r'^Pasal\s+([0-9]+[A-Z]?)$', re.IGNORECASE)
RE_PASAL_TRAILING = re.compile(r'^Pasal\s+([0-9]+[A-Z]?)\s+(.*)$', re.IGNORECASE)  # capture trailing text
RE_AYAT = re.compile(r'^\(\s*(\d+)\s*\)\s*(.*)$')  # starts with (n) followed by text (text optional)
RE_HURUF = re.compile(r'^[a-z]\.\s*(.*)$', re.IGNORECASE)

def is_bab_declare(line):
    return bool(RE_BAB.fullmatch(line.strip()))

def get_bab_number(line):
    m = RE_BAB.fullmatch(line.strip())
    return m.group(1) if m else None

def is_pasal_declare(line):
    return bool(RE_PASAL.fullmatch(line.strip()))

def get_pasal_number(line):
    m = RE_PASAL.fullmatch(line.strip())
    return m.group(1) if m else None

def trailing_after_pasal(line):
    m = RE_PASAL_TRAILING.match(line.strip())
    return m.group(2).strip() if m and m.group(2) else ""

def is_ayat_line(line):
    # Only accept numeric bullets at start of line like "(1) text"
    # Reject lines that contain the word 'ayat' (those are references, not bullets)
    if 'ayat' in line.lower():
        return None
    m = RE_AYAT.match(line)
    return m

# parser for UU2008 and PP2019 
def parse_uu_like(source_name, path, start_marker='BAB I', stop_on_penjelasan=True):
    lines = extract_lines_from_pdf(path)
    rows = []  # list of dicts: source,bab,pasal,ayat,text
    started = False
    current_bab = ""
    current_pasal = ""
    current_ayat = None
    pasal_buffer = []     # lines collected at pasal-level before any numeric ayat seen
    ayat_buffer = []      # lines collected for current numeric ayat
    found_ayat_in_pasal = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # stop when we hit penjelasan (tolerant)
        if stop_on_penjelasan:
            if 'PENJELASAN' in line.upper():
                # stop parsing this doc entirely
                break

        # wait until explicit start marker (e.g., "BAB I")
        if not started:
            if line.strip().upper() == start_marker.upper():
                started = True
                # if that line exactly is BAB I, capture if needed
                if is_bab_declare(line):
                    current_bab = get_bab_number(line)
                i += 1
                continue
            else:
                i += 1
                continue

        # detect BAB declaration EXACT (single-line)
        if is_bab_declare(line):
            # flush previous pasal (if any)
            if current_pasal:
                if current_ayat and ayat_buffer:
                    # flush current ayat first
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
                    ayat_buffer = []
                # implicit pasal -> if no ayat seen in pasal then combine pasal_buffer into ayat=1
                if not found_ayat_in_pasal and pasal_buffer:
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": "1", "text": " ".join([ln for ln in pasal_buffer if ln != "<PAGE_BREAK>"]).strip()})
                pasal_buffer = []
                found_ayat_in_pasal = False
                current_ayat = None
            # set new bab
            current_bab = get_bab_number(line)
            i += 1
            continue

        # detect Pasal declaration EXACT (no other text) OR with trailing text -> keep trailing
        if is_pasal_declare(line):
            # flush previous pasal (same as above)
            if current_pasal:
                if current_ayat and ayat_buffer:
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
                    ayat_buffer = []
                if not found_ayat_in_pasal and pasal_buffer:
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": "1", "text": " ".join([ln for ln in pasal_buffer if ln != "<PAGE_BREAK>"]).strip()})
                pasal_buffer = []
                found_ayat_in_pasal = False
                current_ayat = None
            # set new pasal
            current_pasal = get_pasal_number(line)
            # capture trailing text after "Pasal X ..." if present on same line
            tail = trailing_after_pasal(line)
            if tail:
                pasal_buffer.append(tail)
            i += 1
            continue

        # detect ayat lines (strict)
        m = is_ayat_line(line)
        if m and current_pasal:
            # flush prior ayat buffer
            if current_ayat and ayat_buffer:
                rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
                ayat_buffer = []
            current_ayat = m.group(1)
            found_ayat_in_pasal = True
            rest = m.group(2).strip()
            if rest:
                ayat_buffer.append(rest)
            i += 1
            continue

        # huruf (a., b.) or normal text:
        if current_pasal:
            # if inside an ayat, append to ayat buffer
            if current_ayat:
                ayat_buffer.append(line)
            else:
                # no numeric ayat yet, keep as pasal-level buffer
                pasal_buffer.append(line)
            i += 1
            continue

        # otherwise ignore line
        i += 1

    # end while - flush end-of-doc buffers
    if current_pasal:
        if current_ayat and ayat_buffer:
            rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
        elif not found_ayat_in_pasal and pasal_buffer:
            rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": "1", "text": " ".join([ln for ln in pasal_buffer if ln != "<PAGE_BREAK>"]).strip()})
    return rows

# parser for UU2024 (special rules) 
def parse_uu2024(source_name, path):
    lines = extract_lines_from_pdf(path)
    rows = []
    started = False
    current_pasal = ""
    current_bab = ""
    current_ayat = None
    pasal_buffer = []
    ayat_buffer = []
    found_ayat_in_pasal = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # stop marker exact-ish
        if 'PENJELASAN UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 1 TAHUN 2024' in line.upper():
            break

        # start once we see "Pasal 1" (user-specified)
        if not started:
            if line.strip().startswith('Pasal 1'):
                started = True
                # if line is quoted style, parser still will detect later
            else:
                i += 1
                continue

        # detect quoted pasal declarations: must begin with a quote char before "Pasal"
        if re.match(r'^[\'"“”]\s*Pasal\s+([0-9A-Z]+)', line.strip(), re.IGNORECASE):
            # flush previous pasal
            if current_pasal:
                if current_ayat and ayat_buffer:
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
                    ayat_buffer = []
                if not found_ayat_in_pasal and pasal_buffer:
                    rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": "1", "text": " ".join([ln for ln in pasal_buffer if ln != "<PAGE_BREAK>"]).strip()})
                pasal_buffer = []
                found_ayat_in_pasal = False
                current_ayat = None
            # set new pasal
            m = re.match(r'^[\'"“”]\s*Pasal\s+([0-9A-Z]+)', line.strip(), re.IGNORECASE)
            current_pasal = m.group(1)
            # trailing text after quoted pasal?
            m2 = re.match(r'^[\'"“”]\s*Pasal\s+([0-9A-Z]+)\s*(.*)$', line.strip(), re.IGNORECASE)
            if m2 and m2.group(2):
                pasal_buffer.append(m2.group(2).strip())
            i += 1
            continue

        # numeric ayat detection same as others (only numeric bullets)
        m = is_ayat_line(line)
        if m and current_pasal:
            if current_ayat and ayat_buffer:
                rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
                ayat_buffer = []
            current_ayat = m.group(1)
            found_ayat_in_pasal = True
            rest = m.group(2).strip()
            if rest:
                ayat_buffer.append(rest)
            i += 1
            continue

        # huruf or normal text
        if current_pasal:
            if current_ayat:
                ayat_buffer.append(line)
            else:
                pasal_buffer.append(line)
            i += 1
            continue

        i += 1

    # flush final
    if current_pasal:
        if current_ayat and ayat_buffer:
            rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": current_ayat, "text": " ".join([ln for ln in ayat_buffer if ln != "<PAGE_BREAK>"]).strip()})
        elif not found_ayat_in_pasal and pasal_buffer:
            rows.append({"source": source_name, "bab": current_bab, "pasal": current_pasal, "ayat": "1", "text": " ".join([ln for ln in pasal_buffer if ln != "<PAGE_BREAK>"]).strip()})
    return rows

#  combine duplicates helper 
def combine_duplicates(rows):
    # maintain insertion order and combine texts for identical (source,bab,pasal,ayat)
    combined = {}
    order = []
    for r in rows:
        key = (r['source'], r['bab'] or "", r['pasal'] or "", str(r['ayat'] or ""))
        if key not in combined:
            combined[key] = r['text']
            order.append(key)
        else:
            # append with a space. also, top row already kept first
            combined[key] = combined[key] + " " + r['text']
    out = []
    for k in order:
        out.append({"source": k[0], "bab": k[1], "pasal": k[2], "ayat": k[3], "text": combined[k]})
    return out

# main 
def main():
    # file names 
    uu2008 = "UU_NO_11_2008-1.PDF"
    pp2019 = "PP_NO_71_2019.PDF"
    uu2024 = "UU_NO_1_2024.pdf"

    all_rows = []

    # UU 11/2008
    try:
        all_rows.extend(parse_uu_like("UU 11/2008", uu2008, start_marker='BAB I', stop_on_penjelasan=True))
    except FileNotFoundError:
        print("Warning: UU2008 file not found:", uu2008)

    # PP 71/2019
    try:
        all_rows.extend(parse_uu_like("PP 71/2019", pp2019, start_marker='BAB I', stop_on_penjelasan=True))
    except FileNotFoundError:
        print("Warning: PP2019 file not found:", pp2019)

    # UU 1/2024 (special)
    try:
        all_rows.extend(parse_uu2024("UU 1/2024", uu2024))
    except FileNotFoundError:
        print("Warning: UU2024 file not found:", uu2024)

    # combine identical rows (same source,bab,pasal,ayat) into single row by concatenating texts
    combined = combine_duplicates(all_rows)

    # write CSV
    if combined:
        df = pd.DataFrame(combined, columns=["source", "bab", "pasal", "ayat", "text"])
        df.to_csv("pasal_master.csv", index=False, encoding="utf-8")
        print("Wrote pasal_master.csv with", len(df), "rows.")
    else:
        # write empty CSV with headers
        df = pd.DataFrame(columns=["source", "bab", "pasal", "ayat", "text"])
        df.to_csv("pasal_master.csv", index=False, encoding="utf-8")
        print("No rows extracted; created empty pasal_master.csv with headers.")

if __name__ == "__main__":
    main()

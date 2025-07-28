import fitz       # PyMuPDF
import re
import numpy as np
import json
import os
from operator import itemgetter

# 1. Utility functions for font stats (from friend's utils.py)
def fonts(doc):
    styles = {}
    font_counts = {}
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        size = round(s["size"], 1)
                        identifier = f"{size}"
                        styles[identifier] = {'size': size, 'font': s['font']}
                        font_counts[identifier] = font_counts.get(identifier, 0) + 1
    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)
    if not font_counts:
        raise ValueError("No fonts found.")
    return font_counts, styles

def font_tags(font_counts, styles):
    p_style = styles[font_counts[0][0]]
    p_size = p_style['size']
    font_sizes = sorted([float(size) for size, _ in font_counts], reverse=True)
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = 'p'
        elif size > p_size:
            size_tag[size] = f"h{idx}"
        else:
            size_tag[size] = f"s{idx}"
    return size_tag, p_size

# 2. Numbering and heuristic heading patterns (from your code)
HEADING_NUMBERING_PATTERNS = [
    re.compile(r"^(\d+)(\.\d+)*\.?\s+"),             # "1.", "1.2.3 "
    re.compile(r"^[IVXLCDM]+\.\s+", re.IGNORECASE),  # "IV."
    re.compile(r"^[A-Z]\.\s+"),                      # "A.", "B."
    re.compile(r"^Phase\s+[IVXLCDM]+:", re.IGNORECASE),  # "Phase I:", "Phase II:"
    re.compile(r"^Phase\s+\d+:", re.IGNORECASE),     # "Phase 1:"
]
DATE_PATTERNS = [
    re.compile(r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s*$", re.I),
    re.compile(r"^\s*\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*$", re.I),
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$"),
]
APPENDIX_PATTERN = re.compile(r"^Appendix\s+[A-Z]:", re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"^What\s+could.*\?$", re.IGNORECASE)


def is_date_text(text):
    return any(p.match(text.strip()) for p in DATE_PATTERNS)

def is_heading_numbered(text):
    return any(pat.match(text.strip()) for pat in HEADING_NUMBERING_PATTERNS)

def get_numbering_depth(text):
    text = text.strip()
    if APPENDIX_PATTERN.match(text): return 2
    m = re.match(r"^(\d+(\.\d+)*)(\.?)\s+", text)
    if m: return min(text.count('.')+1, 3)
    if re.match(r"^[IVXLCDM]+\.\s+", text, re.I): return 1
    if re.match(r"^[A-Z]\.\s+", text): return 1
    return None

def is_contextual_heading(text):
    text = text.strip().lower()
    keywords = ["summary", "background", "approach", "requirements", "evaluation",
                "timeline", "milestones", "terms of reference", "membership", "criteria",
                "process", "chair", "meetings", "accountability", "policies", "preamble",
                "funding", "phases", "resources"]
    return any(kw in text for kw in keywords) or bool(QUESTION_PATTERN.match(text))

# 3. Final outline extraction (fusion logic)
def extract_outline_final(doc):
    font_counts, styles = fonts(doc)
    size_tag, p_size = font_tags(font_counts, styles)
    headings = []
    seen = set()
    # Get all text elements for global context (for your code's logic)
    all_elements = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        size = round(s["size"], 1)
                        tag = size_tag.get(size)
                        text = s["text"].strip()
                        if not text or len(text) < 2 or is_date_text(text):
                            continue
                        fonts_ = s["font"]
                        is_bold = any("bold" in fonts_.lower() for fonts_ in [s['font']])
                        is_italic = any("italic" in fonts_.lower() or "oblique" in fonts_.lower() for fonts_ in [s['font']])
                        bbox = s["bbox"]
                        indent = bbox[0]
                        ypos = bbox[1]
                        all_elements.append({
                            "text": text,
                            "font_size": size,
                            "tag": tag,
                            "font": s['font'],
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "indent": indent,
                            "ypos": ypos,
                            "page": page_num
                        })
    # Run heading detection (friend's + your heuristics)
    font_sizes_seen = [s["font_size"] for s in all_elements]
    top4 = sorted(set(font_sizes_seen), reverse=True)[:4]
    for elem in all_elements:
        text, tag, size = elem["text"], elem["tag"], elem["font_size"]
        if (
            not text or len(text) < 3
            or text.lower().startswith("page ")
            or "page" in text.lower()
            or "copyright" in text.lower()
            or "qualifications board" in text.lower()
            or "international" in text.lower()
            or all(c in ".·•—–_" for c in text.replace(" ", ""))
        ):
            continue
        # skip duplicates
        k = (text, elem["page"])
        if k in seen: continue
        # --- heading candidate ---
        is_heading = False
        # (A) Strong font-based ruler
        if tag and tag.startswith("h") and size in top4:
            hlevel = int(tag[1])
            if hlevel <= 4:
                is_heading = True
                base_level = hlevel
        else:
            base_level = None
        # (B) Numbering/context fix (your code)
        numbered = get_numbering_depth(text)
        if numbered is not None:
            is_heading = True
            base_level = min(numbered+1, 4)
        # Contextual fallback
        if not is_heading and is_contextual_heading(text):
            is_heading = True
            base_level = 3  # "H2" or "H3" for context headings
        # Tight filter: Only accept headings up to H3
        if is_heading and base_level is not None and base_level > 1:
            level_map = {2: "H1", 3: "H2", 4: "H3"}
            remapped_level = level_map.get(base_level)
            if remapped_level:
                seen.add(k)
                headings.append({
                    "level": remapped_level, "text": text, "page": elem["page"]
                })
    return headings

# 4. Title detection (friend's logic + metadata)
def get_title_final(doc, size_tag):
    title = doc.metadata.get("title", "").strip() if doc.metadata else ""
    # Valid title?
    if title and title.lower() not in ("untitled", "document", "pdf", "Untitled Document", "untitled document"):
        return title
    # Fallback to largest font h1 on page 1 or 2 (friend's logic)
    candidates = []
    for page_num, page in enumerate(doc, start=1):
        if page_num > 2: break  # restrict to first two pages
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        size = round(s["size"], 1)
                        tag = size_tag.get(size)
                        text = s["text"].strip()
                        if tag == "h1" and len(text) > 5:
                            candidates.append((text, size, page_num))
    if candidates:
        # Largest font first, then earliest
        return sorted(candidates, key=lambda x: (-x[1], x[2]))[0][0]
    return "Untitled Document"

# 5. Main extractor
def extract_pdf_round1a(pdf_path):
    doc = fitz.open(pdf_path)
    font_counts, styles = fonts(doc)
    size_tag, p_size = font_tags(font_counts, styles)
    title = get_title_final(doc, size_tag)
    outline = extract_outline_final(doc)
    return {"title": title, "outline": outline}

# 6. Example script usage for batch processing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adobe India Hackathon 2025 Round 1A PDF Outline Extractor")
    parser.add_argument("input", help="Input PDF file or folder")
    parser.add_argument("output", help="Output JSON file or folder")
    args = parser.parse_args()

    # Decide batch or single
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(args.input, filename)
                res = extract_pdf_round1a(pdf_path)
                out_path = os.path.join(args.output, os.path.splitext(filename)[0] + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"✅ {out_path} ({len(res['outline'])} headings)")
    else:
        res = extract_pdf_round1a(args.input)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {args.output} ({len(res['outline'])} headings)")

import os
import numpy as np
import fitz  # PyMuPDF
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import onnxruntime as ort

# ONNX model is included in Docker image
ONNX_MODEL_PATH = "/app/minilm-sentence-transformer.onnx"

def _load_onnx_session():
    return ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

def _embed_texts(texts, ort_session, max_length=256):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/minilm")
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    # Run ONNX model
    token_embeddings = ort_session.run(None, ort_inputs)[0]  # shape: (batch, seq_len, hidden)
    # Attention mask for mean pooling (batch, seq_len)
    input_mask = inputs["attention_mask"]
    # Convert mask to correct shape/type
    input_mask_expanded = np.expand_dims(input_mask, -1)
    masked_embeddings = token_embeddings * input_mask_expanded
    sum_embeddings = np.sum(masked_embeddings, axis=1)
    n_tokens = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1, a_max=None)
    # Mean pooling
    sentence_embeddings = sum_embeddings / n_tokens
    # Now normalize
    from sklearn.preprocessing import normalize
    return normalize(sentence_embeddings)


def extract_sections_from_outline(pdf_path, outline, min_content_length=30):
    doc = fitz.open(pdf_path)
    # Sections: use headings and group text between them
    lines_per_page = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        lines_per_page[page_num+1] = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
    section_texts = []
    for i, h in enumerate(outline["outline"]):
        start_page = h["page"]
        end_page = (outline["outline"][i+1]["page"]-1
                    if i+1<len(outline["outline"]) else len(doc))
        # Slice text for all relevant pages
        section = []
        for p in range(start_page, end_page+1):
            section.extend(lines_per_page.get(p, []))
        body = "\n".join(section)
        if len(body) > min_content_length:
            section_texts.append({
                "title": h["text"],
                "level": h["level"],
                "page": start_page,
                "body": body
            })
    return section_texts

def extract_persona_keywords(persona, job):
    # Quick heuristics: nouns from persona+job, split on space and hyphens, dedup
    # Advanced: use spaCy noun phrase/chunk extractor if available
    import re
    tokens = re.findall(r'\w+', persona + " " + job)
    # Remove stopwords, deduplicate, lowercase
    blacklist = set(['the', 'and', 'for', 'with', 'to', 'in', 'on', 'a', 'of', 'by', 'is', 'at'])
    keywords = set(t.lower() for t in tokens if len(t)>2 and t.lower() not in blacklist)
    return keywords

def compute_tfidf_scores(section_bodies, persona_job_text):
    # Fit on all section bodies plus persona+job text
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=1)
    X = vectorizer.fit_transform(section_bodies + [persona_job_text])
    persona_vec = X[-1]
    section_vecs = X[:-1]
    # Cosine similarities
    # Normalize manually, just to be safe if sklearn version is old
    sim = (section_vecs * persona_vec.T).toarray().ravel()

    return sim

def extract_persona_insights(pdf_paths, outlines, persona, job, output_dir):
    """
    Args:
      pdf_paths: list of PDF file paths, parallel to outlines
      outlines: list of dicts from round_1a_extractor (output JSON)
      persona: string
      job: string
      output_dir: for report
    Returns:
      structured insight JSON
    """
    # 1. Sectionize documents
    section_list = []
    for pdf_path, outline in zip(pdf_paths, outlines):
        doc_sections = extract_sections_from_outline(pdf_path, outline)
        for sec in doc_sections:
            sec['document'] = os.path.basename(pdf_path)
            section_list.append(sec)
    if not section_list:
        return {"error": "No sections found in documents."}

    # 2. Prepare persona/job embedding vector
    ort_session = _load_onnx_session()
    persona_job_text = f"{persona}. {job}"
    persona_emb = _embed_texts([persona_job_text], ort_session)[0]

    # 3. Embed all section bodies
    sec_bodies = [s["body"] for s in section_list]
    sec_embs = _embed_texts(sec_bodies, ort_session)
    emb_sims = np.dot(sec_embs, persona_emb)  # cosine similarity

    # 4. Compute TF-IDF similarity
    tfidf_sims = compute_tfidf_scores(sec_bodies, persona_job_text)

    # 5. Persona keyword overlap
    keywords = extract_persona_keywords(persona, job)
    key_overlaps = np.array([
        len(set(re.findall(r"\w+", s["body"].lower())).intersection(keywords)) / (1+len(keywords))
        for s in section_list
    ])

    # 6. Page position (earlier sections slightly prioritized)
    page_scores = np.array([1 - (s["page"] - 1) / 100.0 for s in section_list])

    # 7. Weighted scoring
    scores = (
        0.5 * emb_sims +
        0.25 * tfidf_sims +
        0.15 * key_overlaps +
        0.10 * page_scores
    )

    top_k = min(8, len(section_list)//2+1)
    ranked_idxs = np.argsort(scores)[::-1][:top_k]

    extracted_sections = []
    subsection_analysis = []
    for rank, idx in enumerate(ranked_idxs, start=1):
        sec = section_list[idx]
        extracted_sections.append({
            "document": sec["document"],
            "page": sec["page"],
            "section_title": sec["title"],
            "importance_rank": int(rank)
        })
        highlight_text = summarize_section(sec["body"])
        subsection_analysis.append({
            "document": sec["document"],
            "page": sec["page"],
            "refined_text": highlight_text[:350].replace('\n', ' '),
            "importance_rank": int(rank)
        })

    # Meta
    meta = {
        "documents": [os.path.basename(p) for p in pdf_paths],
        "persona": persona,
        "job": job,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    }

    return {
        "metadata": meta,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def summarize_section(text):
    # Simply extract 1–3 most relevant sentences (future: add offline summarizer)
    # Take first 2 long sentences heuristically
    sents = re.split(r'(?<=[.!?])\s+', text)
    long_sents = [s for s in sents if len(s) > 40]
    return " ".join(long_sents[:2]) if long_sents else text[:180]


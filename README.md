# Adobe India Hackathon 2025: Connecting the Dots

## Project Overview

This repository implements the full offline, Dockerized solution for the Adobe India Hackathon 2025 challenge titled **"Connecting the Dots."**

The project addresses two rounds:

- **Round 1A:** Extracts the hierarchical document outline (title, headings with levels H1-H3, and page numbers) from one or more PDFs.
- **Round 1B:** Given 3–10 PDFs plus a persona and a job description, selects and ranks the most relevant sections and subsections to provide persona-driven insights.

---

## Features

- **100% Offline** — no internet connection required during execution.
- **CPU Only** — no GPU dependency.
- **Dockerized** with Linux/AMD64 base image for easy deployment.
- **Dynamic, Robust Heading Extraction** using combined visual/layout/textual heuristics.
- **Semantic Similarity and Ranking** in Round 1B via ONNX MiniLM embeddings.
- **Supports Multilingual PDFs** and various document layouts.
- **Built-in handling of corner cases** like date exclusion, repeated headers/footers, and form label filtering.

---

## Repository Structure

```
hackathon_solution/
├── main.py                 # Entry point supporting both stages
├── round_1a_extractor.py   # Round 1A: Outline extraction logic
├── round_1b_extractor.py   # Round 1B: Persona-based insight extraction
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build specification
├── minilm-sentence-transformer.onnx  # ONNX embedding model (vendored)
├── models/
│   └── minilm/             # Local tokenizer files for offline usage
├── input/                  # Place your input PDFs here (bind mount)
├── output/                 # Output JSON files will be saved here (bind mount)
├── .gitignore              # Ignored files (cache, temp, models)
└── README.md               # This document
```

---

## Setup Instructions

### 1. Build the Docker Image

From the project root directory, run:

```
docker build -t pdfdots:latest .
```

This installs all dependencies (PyMuPDF, NumPy, ONNX Runtime, Transformers tokenizer) and bundles the MiniLM ONNX model and tokenizer locally.

### 2. Prepare Input Files

Place your input PDFs inside the `input/` directory (or bind mount any folder at runtime).

### 3. Run Round 1A: Document Outline Extraction

```
docker run --rm \
  -v "$(pwd)/input":/app/input \
  -v "$(pwd)/output":/app/output \
  --network none \
  pdfdots:latest stage1
```

- Processes all PDFs in `/app/input/`.
- Produces `.json` outline files in `/app/output/`.
- Works offline and returns hierarchical outlines with title and headings.

### 4. Run Round 1B: Persona-Based Insight Extraction

```
docker run --rm \
  -v "$(pwd)/input":/app/input \
  -v "$(pwd)/output":/app/output \
  --network none \
  pdfdots:latest stage2 \
  --persona "PhD Researcher in Computational Biology" \
  --job "Write a literature review on GNNs for Drug Discovery"
```

- Automatically loads Round 1A JSON outlines or runs extraction if missing.
- Embeds sections and scores their relevance against the persona/job prompt.
- Outputs a detailed ranked JSON insight file at `/app/output/persona_insights.json`.

---

## Input/Output Formats

### Round 1A Output JSON Schema

```
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Main Heading", "page": 1 },
    { "level": "H2", "text": "Subheading", "page": 3 },
    { "level": "H3", "text": "Sub-subheading", "page": 5 }
  ]
}
```

### Round 1B Output JSON Schema

```
{
  "metadata": {
    "documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job": "Write a literature review on GNNs for Drug Discovery",
    "timestamp": "2025-07-24T12:23:13"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page": 3,
      "section_title": "Graph Neural Networks",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc2.pdf",
      "page": 4,
      "refined_text": "This section discusses benchmark datasets used for GNNs.",
      "importance_rank": 2
    }
  ]
}
```

---

## Dependencies

- Python 3.11+
- PyMuPDF
- NumPy
- ONNX Runtime
- Transformers (only tokenizer, offline-vendored)
- Scikit-learn

All dependencies are installed inside Docker without internet access during runtime.

---

## How It Works — Brief Overview

- **Round 1A** uses PyMuPDF to parse PDF layout and text spans, extracts font sizes, styles, and positions.
- Applies heuristics and pattern recognition to detect headings, including numbering patterns, font size ranking, indentation, visual emphasis (bold/italic), colon-ended line handling, and date exclusion.
- Title is picked directly from PDF metadata if available, else heuristically from first page lines.

- **Round 1B** loads these outlines to split documents into sections.
- Uses ONNX MiniLM embeddings for fast CPU semantic encoding.
- Combines cosine similarity, TF-IDF, keyword overlap (extracted from persona/job), and page position for relevance scoring.
- Ranks and outputs best sections and summary highlights.

---

## Notes

- The entire pipeline is **offline** and works with zero internet dependencies.
- The Docker image supports Linux amd64 but can run on Windows/macOS with Docker Desktop.
- Modify or extend tokenizer/model paths as needed if updating models.
- Input and output folder bindings (`-v`) are critical for data movement.

---

## Troubleshooting

- Ensure `minilm-sentence-transformer.onnx` and `models/minilm` tokenizer files are present and copied into Docker image.
- Use entrypoint commands `stage1` or `stage2` with proper flags.
- Check that input PDFs are placed correctly and not empty.
- See logs printed by Docker container for diagnostic messages.

---

## Contact

For queries, feature requests, or bug reports, please reach out to:

HackSmiths

Email: kapilrawat285@gmail.com, 

GitHub: https://github.com/Kapil-RwT/AdobeHackathonPdfExtractor

---

Thank you for checking out this solution! We hope it serves you well in the Adobe India Hackathon.

---

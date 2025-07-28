# import os
# import argparse
# import glob
# import sys
# import json
# from datetime import datetime

# from round_1a_extractor import extract_outline_from_pdf

# def run_round1a(input_dir, output_dir):
#     print("[INFO] Running Round 1A (Outline Extraction)")
#     os.makedirs(output_dir, exist_ok=True)
#     pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")) + glob.glob(os.path.join(input_dir, "*.PDF")))
#     if not pdfs:
#         print(f"No PDFs found in {input_dir}")
#         return
#     for pdf_path in pdfs:
#         try:
#             result = extract_outline_from_pdf(pdf_path)
#             fname = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
#             out_path = os.path.join(output_dir, fname)
#             with open(out_path, "w", encoding="utf-8") as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)
#             print(f"Extracted outline: {os.path.basename(pdf_path)}")
#         except Exception as e:
#             print(f"[ERROR][1A] {pdf_path} --> {e}")

# def run_round1b(input_dir, output_dir, persona, job):
#     print("[INFO] Running Round 1B (Persona-based Extraction)")
#     from round_1b_extractor import extract_persona_insights
#     os.makedirs(output_dir, exist_ok=True)
#     pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")) + glob.glob(os.path.join(input_dir, "*.PDF")))
#     if not pdfs:
#         print(f"No PDFs found in {input_dir}")
#         return
#     outlines = []
#     for pdf_path in pdfs:
#         json_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
#         json_path = os.path.join(output_dir, json_name)
#         if not os.path.exists(json_path):
#             outlines.append(extract_outline_from_pdf(pdf_path))
#         else:
#             with open(json_path, encoding="utf-8") as f:
#                 outlines.append(json.load(f))
#     output_json = extract_persona_insights(
#         pdfs, outlines, persona, job, output_dir
#     )
#     out_report = os.path.join(output_dir, "persona_insights.json")
#     with open(out_report, "w", encoding="utf-8") as f:
#         json.dump(output_json, f, ensure_ascii=False, indent=2)
#     print("Wrote persona insights to", out_report)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('stage', choices=['stage1', 'stage2'], help="stage1: Outline; stage2: Persona insights")
#     parser.add_argument('--input', default='/app/input')
#     parser.add_argument('--output', default='/app/output')
#     parser.add_argument('--persona', type=str, default="", help="Persona description (stage2 only)")
#     parser.add_argument('--job', type=str, default="", help="Job to be done (stage2 only)")
#     args = parser.parse_args()

#     if args.stage == "stage1":
#         run_round1a(args.input, args.output)
#     elif args.stage == "stage2":
#         if not args.persona or not args.job:
#             print("ERROR: --persona and --job must be specified for stage2")
#             sys.exit(1)
#         run_round1b(args.input, args.output, args.persona, args.job)

# if __name__ == "__main__":
#     main()
import os
import argparse
import glob
import sys
import json

# Use new merged logic for round 1a extraction
# from round1a_final import extract_pdf_round1a
from round_1a_extractor import extract_pdf_round1a

def run_round1a(input_dir, output_dir):
    print("[INFO] Running Round 1A (Outline Extraction)")
    os.makedirs(output_dir, exist_ok=True)
    pdfs = sorted(
        glob.glob(os.path.join(input_dir, "*.pdf")) +
        glob.glob(os.path.join(input_dir, "*.PDF"))
    )
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return
    for pdf_path in pdfs:
        try:
            # Call the new logic for each PDF
            result = extract_pdf_round1a(pdf_path)
            fname = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
            out_path = os.path.join(output_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Extracted outline: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"[ERROR][1A] {pdf_path} --> {e}")


def run_round1b(input_dir, output_dir, persona, job):
    print("[INFO] Running Round 1B (Persona-based Extraction)")
    from round_1b_extractor import extract_persona_insights
    os.makedirs(output_dir, exist_ok=True)
    pdfs = sorted(
        glob.glob(os.path.join(input_dir, "*.pdf")) +
        glob.glob(os.path.join(input_dir, "*.PDF"))
    )
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return
    outlines = []
    for pdf_path in pdfs:
        json_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
        json_path = os.path.join(output_dir, json_name)
        if not os.path.exists(json_path):
            outlines.append(extract_pdf_round1a(pdf_path))
        else:
            with open(json_path, encoding="utf-8") as f:
                outlines.append(json.load(f))
    output_json = extract_persona_insights(
        pdfs, outlines, persona, job, output_dir
    )
    out_report = os.path.join(output_dir, "persona_insights.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print("Wrote persona insights to", out_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['stage1', 'stage2'], help="stage1: Outline; stage2: Persona insights")
    parser.add_argument('--input', default='/app/input')
    parser.add_argument('--output', default='/app/output')
    parser.add_argument('--persona', type=str, default="", help="Persona description (stage2 only)")
    parser.add_argument('--job', type=str, default="", help="Job to be done (stage2 only)")
    args = parser.parse_args()

    if args.stage == "stage1":
        run_round1a(args.input, args.output)
    elif args.stage == "stage2":
        if not args.persona or not args.job:
            print("ERROR: --persona and --job must be specified for stage2")
            sys.exit(1)
        run_round1b(args.input, args.output, args.persona, args.job)

if __name__ == "__main__":
    main()

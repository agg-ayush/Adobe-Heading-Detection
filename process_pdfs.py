import os
import json
from pathlib import Path
import fitz  # PyMuPDF

def extract_structure(pdf_path):
    doc = fitz.open(pdf_path)
    result = {"title": "Untitled Document", "outline": []}
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block["type"] == 0:
                for line in block.get("lines", []):
                    text = " ".join(span.get("text", "").strip() for span in line.get("spans", []))
                    if text and len(text.split()) > 2:
                        result["outline"].append({
                            "level": "H1",
                            "text": text.strip(),
                            "page": page_num + 1
                        })
    doc.close()
    if result["outline"]:
        result["title"] = result["outline"][0]["text"]
    return result

def process_pdfs():
    input_dir = Path("dataset/pdfs")
    output_dir = Path("dataset/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        result = extract_structure(str(pdf_file))
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
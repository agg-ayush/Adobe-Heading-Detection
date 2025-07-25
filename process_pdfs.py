import json
from pathlib import Path
from extract_structure import PDFStructureExtractor  # Import the actual model-backed class

def process_pdfs():
    input_dir = Path("dataset/pdfs")
    output_dir = Path("dataset/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PDFStructureExtractor()  # Instantiate the model-backed extractor

    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        result = extractor.extract_structure(str(pdf_file))  # Use the proper function
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    print("Starting processing PDFs")
    process_pdfs()
    print("Completed processing PDFs")

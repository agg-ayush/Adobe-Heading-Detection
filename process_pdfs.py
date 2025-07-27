import json
import os
from pathlib import Path
from extract_structure import PDFStructureExtractor  # Import the actual model-backed class

def process_pdfs():
    # Use Docker mount points as specified in requirements
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PDFStructureExtractor()  # Instantiate the model-backed extractor

    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        return
        
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            result = extractor.extract_structure(str(pdf_file))  # Use the proper function
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Processed {pdf_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    print("Starting processing PDFs")
    process_pdfs()
    print("Completed processing PDFs")

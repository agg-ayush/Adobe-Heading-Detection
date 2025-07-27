# GLAM PDF Document Structure Extractor

A high-performance Docker solution that extracts document structure (title and hierarchical headings) from PDF files using the GLAM (Graph-based Layout Analysis Model).

## Features

✅ **Fast Processing**: Processes multiple PDFs efficiently  
✅ **CPU-Only**: Runs on AMD64 systems without GPU requirements  
✅ **Offline Operation**: No internet connectivity required  
✅ **Compact Model**: 2.31MB model size (well under 200MB limit)  
✅ **Hierarchical Structure**: Extracts title and H1, H2, H3 headings with page numbers  
✅ **Docker Ready**: Containerized solution for easy deployment  

## Quick Start with Docker

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### Run the Container
```bash
# Create input and output directories
mkdir input output

# Copy your PDF files to the input directory
cp your-pdfs/*.pdf input/

# Process all PDFs
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  mysolutionname:somerandomidentifier
```

### Output
For each `filename.pdf` in the input directory, a corresponding `filename.json` will be created in the output directory containing:
- Document title
- Hierarchical outline with heading levels (H1, H2, H3)
- Page numbers for each heading

## Installation for Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify the model file exists:
```bash
ls models/glam_dln.pt  # Should show the model file (2.31MB)
```

## Usage

### Batch Processing (Recommended)
```bash
# Place your PDFs in the input directory
cp your-pdfs/*.pdf input/

# Process all PDFs
python process_pdfs.py
```

### Single File Processing
```bash
# Extract structure from a single PDF
python extract_structure.py input.pdf output.json

# Process specific number of pages
python extract_structure.py document.pdf result.json --max-pages 30

# Verbose output
python extract_structure.py sample.pdf output.json --verbose
```

### Python API
```python
from extract_structure import PDFStructureExtractor

# Initialize extractor
extractor = PDFStructureExtractor()

# Extract structure
result = extractor.extract_structure("document.pdf")

# Result format:
# {
#   "title": "Document Title",
#   "outline": [
#     {"level": "H1", "text": "Chapter 1", "page": 1},
#     {"level": "H2", "text": "Section 1.1", "page": 2}
#   ]
# }
```

## Output Format

The extractor produces JSON in the exact required format:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## Performance Specifications

| Metric | Specification | Actual Performance |
|--------|---------------|-------------------|
| **Execution Time** | ≤10 seconds (50 pages) | ~3.4 seconds/PDF average |
| **Model Size** | ≤200MB | 2.31MB |
| **Network Access** | None required | ✅ Offline |
| **Runtime** | CPU (AMD64) | ✅ CPU-optimized |
| **Memory** | Efficient usage | ~2-4GB actual usage |

## Architecture

The GLAM (Graph-based Layout Analysis Model) solution works through:

1. **PDF Analysis**: Extracts text elements and spatial relationships using PyMuPDF
2. **Graph Construction**: Creates graph representations of document layout
3. **Neural Classification**: Uses GLAM network to classify elements (titles, headers, text)
4. **Hierarchy Detection**: Analyzes patterns to determine heading levels (H1/H2/H3)
5. **Structure Assembly**: Combines classified elements into hierarchical document outline

## Project Structure

```
glam/
├── extract_structure.py      # Main extraction engine
├── process_pdfs.py           # Batch processing script
├── glam_prepare.py          # GLAM preparation utilities
├── models/
│   └── glam_dln.pt          # Trained GLAM model (2.31MB)
├── core/
│   ├── __init__.py
│   ├── common.py            # Data structures and utilities
│   └── models.py            # GLAM neural network architecture
├── debug/                   # Debug and analysis tools
│   ├── debug_pdf.py
│   └── check_levels.py
├── input/                   # Input PDFs directory
├── output/                  # Generated JSON outputs
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker container definition
└── .dockerignore           # Docker build exclusions
```
## Optimization Features

- **Smart Page Filtering**: Processes up to 300 nodes per page for optimal speed/accuracy balance
- **Confidence Thresholding**: Only processes high-confidence predictions (>40%)
- **Memory Efficiency**: CPU-optimized tensor operations and data structures
- **Background Processing**: Handles large documents efficiently

## Error Handling

- **Missing Files**: Clear error messages for missing PDFs or model files
- **Large Documents**: Graceful handling of complex pages with performance fallbacks
- **Corrupt PDFs**: Robust error handling with meaningful diagnostics
- **Memory Limits**: Automatic optimization to prevent memory overflow

## Troubleshooting

### Performance Issues
- Check available RAM (requires ~2-4GB)
- Verify CPU has sufficient cores
- Large documents (>15MB) may take longer

### Accuracy Issues
- Ensure PDF has clear text structure (not scanned images)
- Check if document has consistent heading formatting
- Verify font size differences between heading levels

### File Issues
```bash
# Check model file
ls -la models/glam_dln.pt

# Verify PDF is readable
python -c "import fitz; print(fitz.Document('your.pdf').page_count)"
```

## Example Output

For a legal document:
```json
{
  "title": "Law No. (6) of 2019",
  "outline": [
    { "level": "H1", "text": "Ownership of Jointly Owned Real Property", "page": 6 },
    { "level": "H2", "text": "Jointly Owned Real Property", "page": 7 },
    { "level": "H2", "text": "Major Project", "page": 7 },
    { "level": "H3", "text": "Hotel Project", "page": 7 }
  ]
}
```

## License

This project is licensed under dual licensing:
- Apache 2.0 License
- MIT License

**Note**: PyMuPDF dependency uses AGPL-3.0 license which may have implications for commercial use.

---

**Production-ready Docker solution!** 🚀

## Acknowledgements

- Jilin Wang, Michael Krumdick, Baojia Tong, Hamima Halim, Maxim Sokolov, Vadym Barda, Delphine Vendryes, and Chris Tanner. "A Graphical Approach to Document Layout Analysis". 2023. arXiv: [2308.02051](https://arxiv.org/abs/2308.02051)

# PDF Document Structure Extractor

A high-performance tool that extracts document structure (title and hierarchical headings) from PDF files using the GLAM (Graph-based Layout Analysis Model).

## Features

✅ **Fast Processing**: ≤10 seconds for 50-page PDFs  
✅ **CPU-Only**: Runs on AMD64 systems (8 CPUs, 16GB RAM)  
✅ **No Internet**: Fully offline operation  
✅ **Compact Model**: <200MB model size  
✅ **Hierarchical Structure**: Extracts H1, H2, H3 headings with page numbers  

## Requirements

- Python 3.8+
- CPU-based execution (no GPU required)
- 16GB RAM (8GB+ recommended)
- No internet connectivity needed

## Installation

1. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The requirements.txt includes only core packages needed for heading detection. Training-specific packages are commented out to keep the installation lightweight.

2. **For training (if needed)**:
   Uncomment the training dependencies in `requirements.txt` and reinstall:
   ```bash
   # Uncomment training packages in requirements.txt, then:
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch, fitz, networkx; print('✅ Dependencies installed successfully!')"
   ```

## Quick Start

1. **Extract document structure (title + outline format)**:
   ```bash
   python document_parser.py "examples/sample.pdf" output.json --max-pages 50
   ```

## Usage

### Command Line

```bash
# Main document structure extraction (title + outline)
python document_parser.py input.pdf output.json

# Process specific number of pages
python document_parser.py document.pdf result.json --max-pages 30

# Verbose output
python document_parser.py sample.pdf output.json --verbose

# Alternative heading detection
python heading_detector.py --pdf "input.pdf" --max-pages 50

# Debug PDF content
python pdf_debugger.py input.pdf
```

### Example Usage

```bash
# Extract structure from example PDFs
python document_parser.py "examples/sample.pdf" output.json --max-pages 50

# Quick heading detection
python heading_detector.py --pdf "examples/document.pdf" --max-pages 20
```

### Python API

```python
from document_parser import PDFStructureExtractor

# Initialize extractor
extractor = PDFStructureExtractor("models/glam_dln.pt")

# Extract structure
result = extractor.extract_structure("document.pdf", max_pages=50)

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
| **Execution Time** | ≤10 seconds (50 pages) | ~2-8 seconds |
| **Model Size** | ≤200MB | ~60MB |
| **Network Access** | None required | ✅ Offline |
| **Runtime** | CPU (AMD64) | ✅ CPU-optimized |
| **Memory** | 8 CPUs, 16GB RAM | ~2-4GB actual usage |

## How It Works

1. **PDF Analysis**: Extracts text elements and their spatial relationships using PyMuPDF
2. **Layout Classification**: Uses GLAM neural network to classify elements as titles, headers, text, etc.
3. **Hierarchy Detection**: Analyzes font sizes, positions, and content patterns to determine H1/H2/H3 levels
4. **Structure Assembly**: Combines classified elements into a hierarchical document outline

## File Structure

```
glam/
├── document_parser.py       # Main extraction script (title + outline)
├── heading_detector.py      # Fast heading detection
├── inference.py            # Core inference functionality
├── pdf_debugger.py         # PDF debugging utilities
├── models/
│   └── glam_dln.pt         # Trained GLAM model weights (~60MB)
├── core/                   # Core GLAM module
│   ├── common.py           # Data structures and utilities
│   └── models.py           # Neural network architecture
├── training/               # Training scripts
├── tests/                  # Test variants
├── requirements.txt        # Python dependencies
└── examples/               # Sample PDF documents
```

## Optimization Features

- **Smart Page Filtering**: Skips overly complex pages (>400 elements)
- **Confidence Thresholding**: Only processes high-confidence predictions (>70%)
- **Memory Efficiency**: Immediate CPU tensor transfers, optimized data structures
- **Early Termination**: Stops processing if time limit approaches
- **Hierarchical Classification**: Font-size based heading level detection

## Error Handling

- **Missing Files**: Clear error messages for missing PDFs or model files
- **Large Documents**: Graceful handling of complex pages with performance fallbacks
- **Corrupt PDFs**: Robust error handling with meaningful diagnostics
- **Memory Limits**: Automatic page skipping to prevent memory overflow

## Troubleshooting

### Performance Issues
- Reduce `max_pages` parameter for faster processing
- Check available RAM (requires ~2-4GB)
- Verify CPU has sufficient cores

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

For a document:
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Overview", "page": 2 },
    { "level": "H2", "text": "Main Content", "page": 3 },
    { "level": "H3", "text": "Sub Section", "page": 4 }
  ]
}
```

## License

Licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Based on the GLAM (Graph-based Layout Analysis Model) research:
- Jilin Wang et al. "A Graphical Approach to Document Layout Analysis". 2023. arXiv: [2308.02051](https://arxiv.org/abs/2308.02051)

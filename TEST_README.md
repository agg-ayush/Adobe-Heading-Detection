# GLAM Model Testing

This directory contains test scripts for the Graph-based Layout Analysis Model (GLAM). The trained model weights should be located at `models/glam_dln.pt`.

## Test Scripts

### 1. `test_glam_simple.py` - Basic Test
A simple test script that verifies the model can load and process a PDF document.

**Usage:**
```bash
python test_glam_simple.py
```

**Features:**
- Tests model loading and inference
- Processes the first page of `examples/pdf/book law.pdf`
- Displays detailed statistics about nodes, edges, and clusters
- Shows class distribution and confidence metrics
- No external visualization dependencies

### 2. `test_glam_model.py` - Advanced Test with Visualization
A comprehensive test script with matplotlib visualization capabilities.

**Usage:**
```bash
python test_glam_model.py
```

**Features:**
- Full model testing with detailed analysis
- Visual output showing detected layout elements
- Saves results to `test_outputs/` directory
- Color-coded visualization of different layout classes
- Processes up to 3 pages of the test PDF

**Requirements:** matplotlib (for visualization)

### 3. `test_glam_batch.py` - Batch Testing
A batch processing script for testing multiple PDF files.

**Usage:**
```bash
# Test single PDF
python test_glam_batch.py --pdf "path/to/document.pdf"

# Test all PDFs in a directory
python test_glam_batch.py --dir "path/to/pdf/directory"

# Custom options
python test_glam_batch.py --dir "examples/pdf" --max-pages 5 --output "my_report.json"
```

**Options:**
- `--model`: Path to model file (default: `models/glam_dln.pt`)
- `--pdf`: Single PDF file to test
- `--dir`: Directory containing PDF files (default: `examples/pdf`)
- `--max-pages`: Maximum pages per PDF to process (default: 3)
- `--output`: Output report file (default: `test_report.json`)

**Features:**
- Batch processing of multiple PDF files
- JSON report generation with detailed statistics
- Summary statistics across all processed documents
- Error handling and progress tracking

## Layout Classes

The GLAM model classifies layout elements into the following categories:

1. **Caption** - Image/table captions
2. **Footnote** - Footnote text
3. **Formula** - Mathematical formulas
4. **List-item** - List items and bullet points
5. **Page-footer** - Page footer content
6. **Page-header** - Page header content
7. **Picture** - Images and figures
8. **Section-header** - Section headings
9. **Table** - Table structures
10. **Text** - Regular body text
11. **Title** - Document titles and main headings

## Output Files

### Simple Test Output
Console output showing:
- Model loading status
- Graph structure statistics
- Node classification results
- Cluster analysis
- Confidence metrics

### Advanced Test Output
- Console logs (same as simple test)
- `test_outputs/page_X_analysis.png` - Visual analysis for each page

### Batch Test Output
- `test_report.json` - Comprehensive JSON report with:
  - Overall statistics
  - Per-PDF results
  - Per-page details
  - Class distribution across all documents
  - Performance metrics

## Example JSON Report Structure

```json
{
  "timestamp": "2025-01-15 10:30:00",
  "model_path": "models/glam_dln.pt",
  "device": "cuda",
  "total_pdfs": 1,
  "successful_pdfs": 1,
  "failed_pdfs": 0,
  "overall_statistics": {
    "total_pages_processed": 3,
    "avg_inference_time_per_page": 0.125,
    "overall_class_distribution": {
      "Text": 245,
      "Title": 12,
      "Section-header": 8,
      "Picture": 3
    }
  },
  "results": [...]
}
```

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `models/glam_dln.pt` exists
   - Check file path and permissions

2. **PDF file not found**
   - Verify PDF exists at `examples/pdf/book law.pdf`
   - Check file permissions

3. **CUDA out of memory**
   - The model will automatically fall back to CPU
   - Reduce `max_pages` for large documents

4. **Import errors**
   - Ensure all requirements are installed
   - Check that `GLAM` module is in Python path

5. **Visualization errors** (advanced test only)
   - Install matplotlib: `pip install matplotlib`
   - Or use the simple test script instead

### Performance Notes

- **GPU vs CPU**: The model runs faster on GPU but works fine on CPU
- **Memory usage**: Each page uses ~50-200MB depending on complexity
- **Inference time**: Typically 0.1-0.5 seconds per page on modern hardware

## Requirements

All test scripts require the packages listed in `requirements.txt`. Additional dependencies:

- For visualization: `matplotlib`
- For advanced features: `networkx`, `shapely`

## Model Information

- **Architecture**: Graph Neural Network with node and edge classification
- **Input**: PDF text spans and images converted to graph nodes
- **Output**: Layout element classes and spatial relationships
- **Training data**: DocLayNet dataset (not included)

For more information about the model architecture, see the main README.md file.

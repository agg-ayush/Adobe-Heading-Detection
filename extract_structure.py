#!/usr/bin/env python3
"""
PDF Document Structure Extractor using GLAM Model
Extracts title and headings (H1, H2, H3) from PDF documents.

Requirements:
- Accepts PDF files up to 50 pages
- Extracts title and hierarchical headings
- Outputs JSON in specified format
- Runs in ≤10 seconds for 50-page PDF
- CPU-only operation
- No internet access required
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np
from torch_geometric.data import Data

import fitz  # PyMuPDF

# Import GLAM modules
from GLAM.common import PageEdges, ImageNode, TextNode, PageNodes
from GLAM.models import GLAMGraphNetwork
from dln_glam_prepare import CLASSES_MAP

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce verbosity for production
logger = logging.getLogger(__name__)

INVALID_UNICODE = chr(0xFFFD)

# Class mappings for our use case
TITLE_CLASS = 11  # "Title"
SECTION_HEADER_CLASS = 8  # "Section-header"
TEXT_CLASS = 10  # "Text"


class PDFStructureExtractor:
    """Extract document structure from PDF using GLAM model"""
    
    def __init__(self, model_path: str = "models/glam_dln.pt"):
        """Initialize the extractor"""
        self.model_path = model_path
        self.device = "cpu"  # CPU-only as per requirements
        self.model = None
        self.classes_map = CLASSES_MAP
        
        # Performance optimizations for 10-second constraint
        self.max_nodes_per_page = 400  # Balanced for speed vs accuracy
        self.confidence_threshold = 0.7  # Only high-confidence predictions
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load and optimize the GLAM model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = GLAMGraphNetwork(
                node_features_len=PageNodes.features_len,
                edge_feature_len=PageEdges.features_len,
                initial_hidden_len=512,
                node_classes_len=len(self.classes_map)
            )
            
            # Load model weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            for param in self.model.parameters():
                param.requires_grad_(False)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _extract_page_elements(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract elements from a single page"""
        try:
            page_nodes = PageNodes()
            
            # Extract text and structure from page
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            
            for block in page_dict.get("blocks", []):
                if block["type"] == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            
                            # Skip empty or very short text
                            if len(text) < 2:
                                continue
                            
                            # Clean invalid unicode
                            if INVALID_UNICODE in text:
                                text = text.replace(INVALID_UNICODE, " ")
                            
                            page_nodes.append(TextNode.from_span(span, text=text))
            
            # Skip pages that are too complex (performance constraint)
            if len(page_nodes) > self.max_nodes_per_page:
                logger.warning(f"Skipping page {page_num} - too complex ({len(page_nodes)} nodes)")
                return []
            
            if len(page_nodes) == 0:
                return []
            
            # Create graph structure
            page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)
            
            # Convert to tensors
            node_features = page_nodes.to_node_features()
            edge_index = page_edges.to_edge_index().t()
            edge_features = page_edges.to_edge_features()
            
            if edge_index.shape[0] == 0:
                return []
            
            # Run model inference
            data = Data(
                node_features=node_features.to(self.device),
                edge_index=edge_index.to(self.device),
                edge_features=edge_features.to(self.device),
            )
            
            with torch.no_grad():
                node_class_scores, _ = self.model(data)
            
            # Process results
            node_class_scores = node_class_scores.cpu()
            node_predictions = torch.argmax(node_class_scores, dim=1)
            node_confidences = torch.softmax(node_class_scores, dim=1).max(dim=1)[0]
            
            # Extract relevant elements (titles and headers)
            elements = []
            for i, (node, pred, conf) in enumerate(zip(page_nodes, node_predictions, node_confidences)):
                pred_class = pred.item()
                confidence = conf.item()
                
                # Only process high-confidence title and section headers
                if confidence >= self.confidence_threshold and pred_class in [TITLE_CLASS, SECTION_HEADER_CLASS]:
                    elements.append({
                        "text": node.text,
                        "class": pred_class,
                        "confidence": confidence,
                        "bbox": [node.bbox_min_x, node.bbox_min_y, node.bbox_max_x, node.bbox_max_y],
                        "font_size": node.font_size,
                        "page": page_num
                    })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return []
    
    def _determine_heading_levels(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine H1, H2, H3 levels based on font size and content analysis"""
        if not headers:
            return []
        
        # Sort by font size (largest first) to determine hierarchy
        headers_by_size = sorted(headers, key=lambda x: x["font_size"], reverse=True)
        
        # Get unique font sizes
        font_sizes = sorted(list(set(h["font_size"] for h in headers)), reverse=True)
        
        # Assign heading levels based on font size hierarchy
        level_map = {}
        for i, size in enumerate(font_sizes[:3]):  # Only H1, H2, H3
            if i == 0:
                level_map[size] = "H1"
            elif i == 1:
                level_map[size] = "H2"
            elif i == 2:
                level_map[size] = "H3"
        
        # Additional heuristics for better level detection
        results = []
        for header in headers:
            font_size = header["font_size"]
            text = header["text"]
            
            # Determine level
            level = level_map.get(font_size)
            
            if level is None:
                # Fallback: smaller fonts are likely H3
                if font_size < min(font_sizes):
                    level = "H3"
                else:
                    continue  # Skip if we can't determine level
            
            # Content-based refinements
            if len(text) > 100:  # Very long text is likely not a heading
                continue
            
            # Common heading patterns
            text_lower = text.lower()
            if any(word in text_lower for word in ["chapter", "section", "introduction", "conclusion"]):
                if level == "H2" or level == "H3":
                    level = "H1"  # Promote important sections
            
            results.append({
                "level": level,
                "text": text.strip(),
                "page": header["page"],
                "confidence": header["confidence"],
                "font_size": font_size
            })
        
        # Sort by page number and font size
        results.sort(key=lambda x: (x["page"], -x["font_size"]))
        
        return results
    
    def _extract_title(self, all_elements: List[Dict[str, Any]]) -> str:
        """Extract the document title"""
        # Look for elements classified as "Title"
        titles = [elem for elem in all_elements if elem["class"] == TITLE_CLASS]
        
        if titles:
            # Choose the title with highest confidence on the earliest page
            best_title = max(titles, key=lambda x: (x["confidence"], -x["page"]))
            return best_title["text"].strip()
        
        # Fallback: Look for large text on first few pages that could be a title
        early_headers = [elem for elem in all_elements 
                        if elem["page"] <= 3 and elem["class"] == SECTION_HEADER_CLASS]
        
        if early_headers:
            # Get the largest text from early pages
            largest_header = max(early_headers, key=lambda x: x["font_size"])
            return largest_header["text"].strip()
        
        return "Untitled Document"
    
    def extract_structure(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """Extract document structure from PDF"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        start_time = time.time()
        
        try:
            doc = fitz.Document(pdf_path)
            total_pages = len(doc)
            pages_to_process = min(max_pages, total_pages)
            
            logger.info(f"Processing {pages_to_process} pages from {pdf_path}")
            
            all_elements = []
            
            # Process each page
            for page_num in range(pages_to_process):
                page = doc[page_num]
                elements = self._extract_page_elements(page, page_num + 1)
                all_elements.extend(elements)
                
                # Performance check - abort if taking too long
                elapsed = time.time() - start_time
                if elapsed > 8:  # Leave 2 seconds buffer for post-processing
                    logger.warning(f"Time limit approaching, stopping at page {page_num + 1}")
                    break
            
            doc.close()
            
            # Extract title
            title = self._extract_title(all_elements)
            
            # Extract and classify headers
            headers = [elem for elem in all_elements if elem["class"] == SECTION_HEADER_CLASS]
            structured_headers = self._determine_heading_levels(headers)
            
            # Create output structure
            result = {
                "title": title,
                "outline": [
                    {
                        "level": header["level"],
                        "text": header["text"],
                        "page": header["page"]
                    }
                    for header in structured_headers
                ]
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(f"Found title: '{title}'")
            logger.info(f"Found {len(result['outline'])} headings")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Extract document structure from PDF using GLAM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_structure.py input.pdf output.json
  python extract_structure.py document.pdf result.json --max-pages 30
        """
    )
    
    parser.add_argument("input_pdf", help="Input PDF file path")
    parser.add_argument("output_json", help="Output JSON file path")
    parser.add_argument("--max-pages", type=int, default=50, 
                       help="Maximum pages to process (default: 50)")
    parser.add_argument("--model", default="models/glam_dln.pt",
                       help="Path to GLAM model file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Validate inputs
        if not os.path.exists(args.input_pdf):
            print(f"Error: Input PDF file not found: {args.input_pdf}")
            sys.exit(1)
        
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize extractor and process PDF
        print(f"Initializing GLAM model...")
        extractor = PDFStructureExtractor(args.model)
        
        print(f"Processing PDF: {args.input_pdf}")
        result = extractor.extract_structure(args.input_pdf, args.max_pages)
        
        # Save result
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Structure extracted successfully!")
        print(f"📄 Title: {result['title']}")
        print(f"📋 Found {len(result['outline'])} headings")
        print(f"💾 Output saved to: {args.output_json}")
        
        # Show outline preview
        if result['outline']:
            print("\\n📖 Document Outline:")
            for item in result['outline'][:10]:  # Show first 10
                indent = "  " * (int(item['level'][1]) - 1)
                print(f"   {indent}{item['level']}: {item['text']} (page {item['page']})")
            
            if len(result['outline']) > 10:
                print(f"   ... and {len(result['outline']) - 10} more headings")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

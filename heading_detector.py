#!/usr/bin/env python3
"""
Simple optimized testing script for GLAM model without torch.compile
Uses basic optimizations that work on all platforms.
"""

import logging
import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np

import fitz  # PyMuPDF

# Import GLAM modules
from core.common import PageEdges, ImageNode, TextNode, PageNodes
from core.models import GLAMGraphNetwork
from glam_classes import CLASSES_MAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INVALID_UNICODE = chr(0xFFFD)


class SimpleOptimizedGLAMTester:
    """Simple optimized tester with basic performance improvements"""
    
    def __init__(self, model_path: str = "models/glam_dln.pt"):
        """Initialize the tester"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[GLAMGraphNetwork] = None
        self.classes_map = CLASSES_MAP
        
        # Optimization settings
        self.max_nodes_per_page = 300  # Skip extremely complex pages
        self.edge_threshold = 0.5
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load and optimize the GLAM model"""
        try:
            self.model = GLAMGraphNetwork(
                node_features_len=PageNodes.features_len,
                edge_feature_len=PageEdges.features_len,
                initial_hidden_len=512,
                node_classes_len=12  # Updated to match checkpoint
            )
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            assert self.model is not None, "Model should be loaded at this point"
            self.model.eval()
            
            # Basic optimizations
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("✅ CUDA optimizations enabled")
                
            # Set to inference mode for potential speedup
            for param in self.model.parameters():
                param.requires_grad_(False)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_page_data_fast(self, page) -> tuple:
        """Fast page data extraction with minimal processing"""
        try:
            page_nodes = PageNodes()
            
            # Use optimized text extraction
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            
            text_blocks = 0
            image_blocks = 0
            
            for block in page_dict.get("blocks", []):
                if block["type"] == 0:  # Text block
                    text_blocks += 1
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            # Fast unicode cleaning
                            if INVALID_UNICODE in text:
                                text = text.replace(INVALID_UNICODE, " ")
                            page_nodes.append(TextNode.from_span(span, text=text))
                            
                elif block["type"] == 1:  # Image block
                    image_blocks += 1
                    page_nodes.append(ImageNode.from_page_block(block))
            
            # Skip extremely complex pages
            if len(page_nodes) > self.max_nodes_per_page:
                logger.warning(f"Skipping page with {len(page_nodes)} nodes (exceeds limit of {self.max_nodes_per_page})")
                return None, None, text_blocks, image_blocks
            
            if len(page_nodes) == 0:
                return None, None, text_blocks, image_blocks
            
            # Create edges efficiently
            page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)
            
            return page_nodes, page_edges, text_blocks, image_blocks
            
        except Exception as e:
            logger.error(f"Error extracting page data: {e}")
            return None, None, 0, 0
    
    def process_page_fast(self, page_num: int, page_nodes: PageNodes, page_edges: PageEdges) -> Dict[str, Any]:
        """Fast page processing with optimizations"""
        try:
            # Convert to tensors efficiently
            node_features = page_nodes.to_node_features()
            edge_index = page_edges.to_edge_index().t()
            edge_features = page_edges.to_edge_features()
            
            if edge_index.shape[0] == 0:
                return {
                    "page_number": page_num,
                    "success": False,
                    "error": "No edges found",
                    "stats": {}
                }
            
            # Create data object
            data = Data(
                node_features=node_features.to(self.device),
                edge_index=edge_index.to(self.device),
                edge_features=edge_features.to(self.device),
            )
            
            # Fast inference
            start_time = time.time()
            with torch.no_grad():
                assert self.model is not None, "Model must be loaded before inference"
                node_class_scores, edge_class_scores = self.model(data)
            inference_time = time.time() - start_time
            
            # Move to CPU for processing
            node_class_scores = node_class_scores.cpu()
            edge_class_scores = edge_class_scores.cpu()
            
            # Fast predictions
            node_predictions = torch.argmax(node_class_scores, dim=1)
            edge_probabilities = torch.sigmoid(edge_class_scores).squeeze()
            
            # Quick edge counting (skip full graph analysis for speed)
            connected_edges = (edge_probabilities >= self.edge_threshold).sum().item()
            
            # Fast class counting
            node_class_counts = {}
            unique_classes, counts = torch.unique(node_predictions, return_counts=True)
            for class_id, count in zip(unique_classes.tolist(), counts.tolist()):
                class_name = self.classes_map.get(class_id, f"Unknown_{class_id}")
                node_class_counts[class_name] = count
            
            # Quick confidence calculation
            node_confidences = torch.softmax(node_class_scores, dim=1).max(dim=1)[0]
            
            return {
                "page_number": page_num,
                "success": True,
                "error": None,
                "stats": {
                    "total_nodes": len(page_nodes),
                    "total_edges": data.edge_index.shape[1] if data.edge_index is not None else 0,
                    "connected_edges": connected_edges,
                    "inference_time": inference_time,
                    "node_class_counts": node_class_counts,
                    "avg_node_confidence": float(node_confidences.mean()),
                    "avg_edge_probability": float(edge_probabilities.mean()),
                }
            }
            
        except Exception as e:
            return {
                "page_number": page_num,
                "success": False,
                "error": str(e),
                "stats": {}
            }
    
    def test_pdf_fast(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """Fast PDF testing with optimizations"""
        pdf_result = {
            "pdf_path": pdf_path,
            "pages": [],
            "summary": {},
            "success": True
        }
        
        try:
            doc = fitz.Document(pdf_path)
            total_pages = len(doc)
            pages_to_process = min(max_pages, total_pages)
            
            logger.info(f"Processing {pages_to_process} pages with simple optimizations")
            
            total_extraction_time = 0
            total_inference_time = 0
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Fast extraction
                extract_start = time.time()
                page_nodes, page_edges, text_blocks, image_blocks = self.extract_page_data_fast(page)
                extraction_time = time.time() - extract_start
                total_extraction_time += extraction_time
                
                if page_nodes is None:
                    pdf_result["pages"].append({
                        "page_number": page_num + 1,
                        "success": False,
                        "error": "No nodes found or page too complex",
                        "stats": {}
                    })
                    continue
                
                # Fast processing
                result = self.process_page_fast(page_num + 1, page_nodes, page_edges)
                total_inference_time += result.get("stats", {}).get("inference_time", 0)
                pdf_result["pages"].append(result)
                
                # Progress every 10 pages
                if (page_num + 1) % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{pages_to_process} pages")
            
            doc.close()
            
            # Calculate summary
            successful_pages = [p for p in pdf_result["pages"] if p["success"]]
            
            if successful_pages:
                total_nodes = sum(p["stats"]["total_nodes"] for p in successful_pages)
                total_connected_edges = sum(p["stats"]["connected_edges"] for p in successful_pages)
                avg_inference_time = total_inference_time / len(successful_pages) if successful_pages else 0
                
                all_class_counts = {}
                for page in successful_pages:
                    for class_name, count in page["stats"]["node_class_counts"].items():
                        all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count
                
                pdf_result["summary"] = {
                    "total_pages": total_pages,
                    "processed_pages": pages_to_process,
                    "successful_pages": len(successful_pages),
                    "failed_pages": pages_to_process - len(successful_pages),
                    "total_nodes": total_nodes,
                    "total_connected_edges": total_connected_edges,
                    "avg_inference_time": avg_inference_time,
                    "total_extraction_time": total_extraction_time,
                    "total_inference_time": total_inference_time,
                    "class_distribution": all_class_counts
                }
            else:
                pdf_result["success"] = False
                pdf_result["summary"] = {"error": "No pages processed successfully"}
                
        except Exception as e:
            pdf_result["success"] = False
            pdf_result["summary"] = {"error": str(e)}
            logger.error(f"Failed to process {pdf_path}: {e}")
        
        return pdf_result


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast heading detection for GLAM model")
    parser.add_argument("--model", default="models/glam_dln.pt", help="Path to model file")
    parser.add_argument("--pdf", default="examples/book law.pdf", help="PDF file to test")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to process")
    parser.add_argument("--output", default="test_report_fast.json", help="Output report file")
    parser.add_argument("--max-nodes", type=int, default=300, help="Max nodes per page")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        sys.exit(1)
    
    try:
        tester = SimpleOptimizedGLAMTester(args.model)
        tester.max_nodes_per_page = args.max_nodes
        
        start_time = time.time()
        result = tester.test_pdf_fast(args.pdf, args.max_pages)
        total_time = time.time() - start_time
        
        # Save report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": args.model,
            "device": tester.device,
            "max_nodes_per_page": tester.max_nodes_per_page,
            "total_processing_time": total_time,
            "result": result
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\\n" + "="*60)
        print("FAST GLAM MODEL TEST SUMMARY")
        print("="*60)
        
        if result["success"] and "summary" in result:
            stats = result["summary"]
            successful_pages = stats.get('successful_pages', 0)
            
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Pages processed: {successful_pages}")
            
            if successful_pages > 0:
                print(f"Average per page: {total_time/successful_pages:.3f}s")
                print(f"Extraction time: {stats.get('total_extraction_time', 0):.2f}s")
                print(f"Inference time: {stats.get('total_inference_time', 0):.2f}s")
                
                # Calculate throughput
                pages_per_second = successful_pages / total_time
                print(f"Throughput: {pages_per_second:.1f} pages/second")
            
            if "class_distribution" in stats:
                total_elements = sum(stats["class_distribution"].values())
                print(f"\\nTotal layout elements detected: {total_elements:,}")
                print("Class distribution:")
                for class_name, count in sorted(stats["class_distribution"].items()):
                    percentage = (count / total_elements) * 100 if total_elements > 0 else 0
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        logger.info(f"✅ Fast testing completed! Report saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Fast testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

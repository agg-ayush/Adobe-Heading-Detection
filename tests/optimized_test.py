#!/usr/bin/env python3
"""
Optimized batch testing script for GLAM model with performance improvements
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Union
import multiprocessing as mp

# Add parent directory to path to import from training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import networkx as nx
from torch_geometric.data import Data, Batch
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


class OptimizedGLAMTester:
    """Optimized batch tester for GLAM model with performance improvements"""
    
    def __init__(self, model_path: str = "models/glam_dln.pt", enable_optimizations: bool = True):
        """Initialize the optimized tester"""
        self.model_path = model_path
        self.device: Any = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.classes_map = CLASSES_MAP
        self.results = []
        self.enable_optimizations = enable_optimizations
        
        # Optimization settings
        self.batch_size = 4 if self.device == "cuda" else 2  # Process multiple pages in batches
        self.max_nodes_per_page = 500  # Skip extremely complex pages
        self.edge_threshold = 0.5
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Optimizations enabled: {enable_optimizations}")
        logger.info(f"Batch size: {self.batch_size}")
        
        self._load_model()
    
    def _load_model(self):
        """Load and optimize the GLAM model"""
        try:
            self.model = GLAMGraphNetwork(
                node_features_len=PageNodes.features_len,
                edge_feature_len=PageEdges.features_len,
                initial_hidden_len=512,
                node_classes_len=len(self.classes_map)
            )
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Enable optimizations
            if self.enable_optimizations:
                # Compile model for faster inference (PyTorch 2.0+)
                try:
                    if hasattr(torch, 'compile'):
                        self.model = torch.compile(self.model, mode='reduce-overhead')
                        logger.info("✅ Model compiled for faster inference")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
                
                # Enable inference optimizations
                torch.backends.cudnn.benchmark = True
                if self.device == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("✅ CUDA optimizations enabled")
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_page_data_fast(self, page) -> tuple:
        """Fast page data extraction with minimal processing"""
        try:
            page_nodes = PageNodes()
            
            # Use faster text extraction method
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            
            text_blocks = 0
            image_blocks = 0
            
            for block in page_dict.get("blocks", []):
                if block["type"] == 0:  # Text block
                    text_blocks += 1
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            if INVALID_UNICODE in text:
                                text = text.replace(INVALID_UNICODE, " ")
                            page_nodes.append(TextNode.from_span(span, text=text))
                            
                elif block["type"] == 1:  # Image block
                    image_blocks += 1
                    page_nodes.append(ImageNode.from_page_block(block))
            
            # Skip extremely complex pages that would slow down processing
            if len(page_nodes) > self.max_nodes_per_page:
                logger.warning(f"Skipping page with {len(page_nodes)} nodes (too complex)")
                return None, None, text_blocks, image_blocks
            
            if len(page_nodes) == 0:
                return None, None, text_blocks, image_blocks
            
            # Create edges more efficiently
            page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)
            
            return page_nodes, page_edges, text_blocks, image_blocks
            
        except Exception as e:
            logger.error(f"Error extracting page data: {e}")
            return None, None, 0, 0
    
    def process_page_batch(self, pages_data: List[tuple]) -> List[Dict[str, Any]]:
        """Process multiple pages in a batch for better GPU utilization"""
        if not pages_data:
            return []
        
        results = []
        valid_data = []
        
        # Prepare batch data
        for page_num, (page_nodes, page_edges) in pages_data:
            if page_nodes is None or len(page_nodes) == 0:
                results.append({
                    "page_number": page_num,
                    "success": False,
                    "error": "No nodes found",
                    "stats": {}
                })
                continue
            
            # Convert to tensors
            node_features = page_nodes.to_node_features()
            edge_index = page_edges.to_edge_index().t()
            edge_features = page_edges.to_edge_features()
            
            if edge_index.shape[0] == 0:
                results.append({
                    "page_number": page_num,
                    "success": False,
                    "error": "No edges found",
                    "stats": {}
                })
                continue
            
            # Create data object
            data = Data(
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
            )
            
            valid_data.append((page_num, data, page_nodes))
        
        if not valid_data:
            return results
        
        # Batch inference for better performance
        try:
            # Create batch
            batch_data: Any = Batch.from_data_list([data for _, data, _ in valid_data])
            batch_data = batch_data.to(self.device)
            
            # Single forward pass for all pages in batch
            start_time = time.time()
            with torch.no_grad():
                assert self.model is not None  # Model should be initialized by now
                node_class_scores, edge_class_scores = self.model(batch_data)
            inference_time = time.time() - start_time
            
            # Process results for each page in batch
            node_ptr = batch_data.ptr
            edge_ptr = torch.zeros(len(valid_data) + 1, dtype=torch.long)
            
            current_edge = 0
            for i in range(len(valid_data)):
                _, data, _ = valid_data[i]
                edge_ptr[i + 1] = edge_ptr[i] + data.edge_index.shape[1]
            
            for i, (page_num, data, page_nodes) in enumerate(valid_data):
                # Extract this page's results from batch
                node_start, node_end = node_ptr[i], node_ptr[i + 1]
                edge_start, edge_end = edge_ptr[i], edge_ptr[i + 1]
                
                page_node_scores = node_class_scores[node_start:node_end].cpu()
                page_edge_scores = edge_class_scores[edge_start:edge_end].cpu()
                
                # Process results quickly
                node_predictions = torch.argmax(page_node_scores, dim=1)
                edge_probabilities = torch.sigmoid(page_edge_scores).squeeze()
                
                # Fast clustering
                connected_edges = (edge_probabilities >= self.edge_threshold).sum().item()
                
                # Quick statistics
                node_class_counts = {}
                for pred in node_predictions:
                    class_id = int(pred.item())  # Convert to int for dictionary lookup
                    class_name = self.classes_map.get(class_id, f"Unknown_{class_id}")
                    node_class_counts[class_name] = node_class_counts.get(class_name, 0) + 1
                
                node_confidences = torch.softmax(page_node_scores, dim=1).max(dim=1)[0]
                
                results.append({
                    "page_number": page_num,
                    "success": True,
                    "error": None,
                    "stats": {
                        "total_nodes": len(page_nodes),
                        "total_edges": data.edge_index.shape[1],
                        "connected_edges": connected_edges,
                        "inference_time": inference_time / len(valid_data),  # Amortized time
                        "node_class_counts": node_class_counts,
                        "avg_node_confidence": float(node_confidences.mean()),
                        "avg_edge_probability": float(edge_probabilities.mean()),
                    }
                })
            
        except Exception as e:
            # Fallback to individual processing
            logger.warning(f"Batch processing failed, falling back to individual: {e}")
            for page_num, data, page_nodes in valid_data:
                result = self._process_single_page(page_num, data, page_nodes)
                results.append(result)
        
        return results
    
    def _process_single_page(self, page_num: int, data: Data, page_nodes: PageNodes) -> Dict[str, Any]:
        """Process a single page (fallback method)"""
        try:
            data = data.to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                assert self.model is not None  # Model should be initialized by now
                node_class_scores, edge_class_scores = self.model(data)
            inference_time = time.time() - start_time
            
            # Process results
            node_class_scores = node_class_scores.cpu()
            edge_class_scores = edge_class_scores.cpu()
            
            node_predictions = torch.argmax(node_class_scores, dim=1)
            edge_probabilities = torch.sigmoid(edge_class_scores).squeeze()
            
            connected_edges = (edge_probabilities >= self.edge_threshold).sum().item()
            
            node_class_counts = {}
            for pred in node_predictions:
                class_id = int(pred.item())  # Convert to int for dictionary lookup
                class_name = self.classes_map.get(class_id, f"Unknown_{class_id}")
                node_class_counts[class_name] = node_class_counts.get(class_name, 0) + 1
            
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
    
    def test_pdf_optimized(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """Optimized PDF testing with batching and performance improvements"""
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
            
            logger.info(f"Processing {pages_to_process} pages with optimizations")
            
            # Extract data for all pages first (can be parallelized)
            all_pages_data = []
            extraction_start = time.time()
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                page_nodes, page_edges, text_blocks, image_blocks = self.extract_page_data_fast(page)
                
                if page_nodes is not None:
                    all_pages_data.append((page_num + 1, (page_nodes, page_edges)))
                else:
                    # Add failed page
                    pdf_result["pages"].append({
                        "page_number": page_num + 1,
                        "success": False,
                        "error": "No nodes found or page too complex",
                        "stats": {}
                    })
            
            extraction_time = time.time() - extraction_start
            logger.info(f"Data extraction completed in {extraction_time:.2f}s")
            
            # Process in batches
            inference_start = time.time()
            for i in range(0, len(all_pages_data), self.batch_size):
                batch = all_pages_data[i:i + self.batch_size]
                batch_results = self.process_page_batch(batch)
                pdf_result["pages"].extend(batch_results)
                
                # Progress logging
                processed = min(i + self.batch_size, len(all_pages_data))
                logger.info(f"Processed batch {i//self.batch_size + 1}: {processed}/{len(all_pages_data)} pages")
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f}s")
            
            doc.close()
            
            # Calculate summary
            successful_pages = [p for p in pdf_result["pages"] if p["success"]]
            
            if successful_pages:
                total_nodes = sum(p["stats"]["total_nodes"] for p in successful_pages)
                total_clusters = sum(p["stats"]["connected_edges"] for p in successful_pages)
                avg_inference_time = sum(p["stats"]["inference_time"] for p in successful_pages) / len(successful_pages)
                
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
                    "total_connected_edges": total_clusters,
                    "avg_inference_time": avg_inference_time,
                    "total_extraction_time": extraction_time,
                    "total_inference_time": inference_time,
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
    """Main function with performance comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized batch test GLAM model")
    parser.add_argument("--model", default="models/glam_dln.pt", help="Path to model file")
    parser.add_argument("--pdf", default="examples/book law.pdf", help="PDF file to test")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to process")
    parser.add_argument("--output", default="test_report_optimized.json", help="Output report file")
    parser.add_argument("--no-optimizations", action="store_true", help="Disable optimizations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        sys.exit(1)
    
    try:
        # Test with optimizations
        tester = OptimizedGLAMTester(
            args.model, 
            enable_optimizations=not args.no_optimizations
        )
        
        if args.batch_size:
            tester.batch_size = args.batch_size
        
        start_time = time.time()
        result = tester.test_pdf_optimized(args.pdf, args.max_pages)
        total_time = time.time() - start_time
        
        # Save report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": args.model,
            "device": tester.device,
            "optimizations_enabled": not args.no_optimizations,
            "batch_size": tester.batch_size,
            "total_processing_time": total_time,
            "result": result
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\\n" + "="*60)
        print("OPTIMIZED GLAM MODEL TEST SUMMARY")
        print("="*60)
        
        if result["success"] and "summary" in result:
            stats = result["summary"]
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Pages processed: {stats.get('successful_pages', 0)}")
            print(f"Average per page: {total_time/stats.get('successful_pages', 1):.3f}s")
            print(f"Extraction time: {stats.get('total_extraction_time', 0):.2f}s")
            print(f"Inference time: {stats.get('total_inference_time', 0):.2f}s")
            
            if "class_distribution" in stats:
                print("\\nClass distribution:")
                for class_name, count in sorted(stats["class_distribution"].items()):
                    print(f"  {class_name}: {count}")
        
        logger.info(f"✅ Optimized testing completed! Report saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Optimized testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

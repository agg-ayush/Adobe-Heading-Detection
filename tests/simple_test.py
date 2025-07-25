#!/usr/bin/env python3
"""
Simple test script for the GLAM model - Core functionality only
This script tests the GLAM model inference without visualization dependencies.
"""

import logging
import os
import sys
import time
from typing import Any, Dict
from typing import List, Tuple, Dict, Any

# Add parent directory to path to import from training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

import fitz  # PyMuPDF

# Import GLAM modules
from core.common import PageEdges, ImageNode, TextNode, get_bytes_per_pixel, PageNodes
from core.models import GLAMGraphNetwork
from glam_classes import CLASSES_MAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INVALID_UNICODE = chr(0xFFFD)


def test_glam_model():
    """Simple test function for the GLAM model"""
    
    # Configuration
    model_path = "models/glam_dln.pt"
    pdf_path = "examples/book law.pdf"
    
    # Check if files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info("Loading GLAM model...")
        model = GLAMGraphNetwork(
            node_features_len=PageNodes.features_len,
            edge_feature_len=PageEdges.features_len,
            initial_hidden_len=512,
            node_classes_len=len(CLASSES_MAP)
        )
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully with {model_params:,} parameters")
        
        # Open PDF and find a page with content
        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.Document(pdf_path)
        
        page_nodes = None
        processed_page_num = None
        
        # Try first few pages to find one with content
        for page_num in range(min(5, len(doc))):
            page: Any = doc[page_num]
            logger.info(f"Trying page {page_num + 1} of {len(doc)}")
            
            # Extract nodes from page
            temp_page_nodes = PageNodes()
        
            page_dict: Dict[str, Any] = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            
            text_blocks = 0
            image_blocks = 0
            
            for block in page_dict["blocks"]:
                if block["type"] == 0:  # Text block
                    text_blocks += 1
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            if INVALID_UNICODE in text:
                                text = text.replace(INVALID_UNICODE, " ")
                            temp_page_nodes.append(TextNode.from_span(span, text=text))
                            
                elif block["type"] == 1:  # Image block
                    image_blocks += 1
                    temp_page_nodes.append(ImageNode.from_page_block(block))
            
            logger.info(f"Page {page_num + 1}: {text_blocks} text blocks, {image_blocks} image blocks, {len(temp_page_nodes)} total nodes")
            
            if len(temp_page_nodes) > 0:
                logger.info(f"✅ Found content on page {page_num + 1}, proceeding with this page")
                page_nodes = temp_page_nodes
                processed_page_num = page_num + 1
                break
            else:
                logger.info(f"Page {page_num + 1} is empty, trying next page...")
        
        if page_nodes is None or len(page_nodes) == 0:
            logger.error("No content found in first 5 pages of PDF")
            return False
        
        logger.info("Extracting page structure...")
        logger.info(f"Processing page {processed_page_num} with {len(page_nodes)} nodes")
        
        # Remove the old page_dict processing since we already did it above
        
        # Create edges
        logger.info("Creating graph edges...")
        page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)
        
        # Convert to tensors
        node_features = page_nodes.to_node_features()
        edge_index = page_edges.to_edge_index().t()
        edge_features = page_edges.to_edge_features()
        
        logger.info(f"Graph structure:")
        logger.info(f"  - Nodes: {node_features.shape[0]}")
        logger.info(f"  - Edges: {edge_index.shape[1]}")
        logger.info(f"  - Node feature dimension: {node_features.shape[1]}")
        logger.info(f"  - Edge feature dimension: {edge_features.shape[1]}")
        
        if edge_index.shape[0] == 0:
            logger.warning("No edges found in the graph")
            return False
        
        # Create data object
        example = Data(
            node_features=node_features.to(device),
            edge_index=edge_index.to(device),
            edge_features=edge_features.to(device),
        )
        
        # Model inference
        logger.info("Running model inference...")
        start_time = time.time()
        
        with torch.no_grad():
            node_class_scores, edge_class_scores = model(example)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f} seconds")
        
        # Process results
        node_class_scores = node_class_scores.cpu()
        edge_class_scores = edge_class_scores.cpu()
        
        # Get predictions
        node_predictions = torch.argmax(node_class_scores, dim=1)
        edge_probabilities = torch.sigmoid(edge_class_scores).squeeze()
        
        # Create graph for clustering
        edge_threshold = 0.5
        graph = nx.Graph()
        
        connected_edges = 0
        for k in range(edge_index.shape[1]):
            src_node = edge_index[0, k].item()
            dst_node = edge_index[1, k].item()
            edge_prob = edge_probabilities[k].item()
            
            if edge_prob >= edge_threshold:
                graph.add_edge(src_node, dst_node, weight=edge_prob)
                connected_edges += 1
            else:
                graph.add_node(src_node)
                graph.add_node(dst_node)
        
        logger.info(f"Connected edges (prob >= {edge_threshold}): {connected_edges}")
        
        # Find clusters
        clusters = list(nx.connected_components(graph))
        logger.info(f"Found {len(clusters)} clusters")
        
        # Analyze results
        logger.info("\\nAnalysis Results:")
        logger.info("-" * 50)
        
        # Node classification statistics
        node_class_counts = {}
        for pred in node_predictions:
            class_id = int(pred.item())  # Convert to int for dictionary lookup
            class_name = CLASSES_MAP.get(class_id, f"Unknown_{class_id}")
            node_class_counts[class_name] = node_class_counts.get(class_name, 0) + 1
        
        logger.info("Node Classifications:")
        for class_name, count in sorted(node_class_counts.items()):
            logger.info(f"  {class_name}: {count} nodes")
        
        # Cluster analysis
        logger.info("\\nCluster Analysis:")
        for i, cluster in enumerate(clusters[:10]):  # Show first 10 clusters
            if len(cluster) > 0:
                # Determine cluster class by majority vote
                cluster_node_scores = node_class_scores[torch.tensor(list(cluster))]
                cluster_class_id = torch.argmax(cluster_node_scores.sum(dim=0)).item()
                cluster_class_name = CLASSES_MAP.get(int(cluster_class_id), f"Unknown_{cluster_class_id}")
                
                logger.info(f"  Cluster {i+1}: {len(cluster)} nodes, class: {cluster_class_name}")
        
        if len(clusters) > 10:
            logger.info(f"  ... and {len(clusters) - 10} more clusters")
        
        # Model confidence analysis
        node_confidences = torch.softmax(node_class_scores, dim=1).max(dim=1)[0]
        edge_confidence = edge_probabilities.mean().item()
        
        logger.info(f"\\nConfidence Analysis:")
        logger.info(f"  Average node confidence: {node_confidences.mean():.3f}")
        logger.info(f"  Min node confidence: {node_confidences.min():.3f}")
        logger.info(f"  Max node confidence: {node_confidences.max():.3f}")
        logger.info(f"  Average edge probability: {edge_confidence:.3f}")
        
        doc.close()
        logger.info("\\nTest completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    logger.info("Starting GLAM model test...")
    logger.info("=" * 60)
    
    success = test_glam_model()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

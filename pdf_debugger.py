#!/usr/bin/env python3
"""
Debug script to investigate PDF parsing issues
"""

import logging
import os
import sys
from typing import Any, Dict, List

import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INVALID_UNICODE = chr(0xFFFD)

def debug_pdf_parsing():
    """Debug PDF parsing to see what's happening"""
    pdf_path = "examples/book law.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    try:
        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.Document(pdf_path)
        logger.info(f"PDF has {len(doc)} pages")
        
        # Check first few pages
        for page_num in range(min(3, len(doc))):
            page: Any = doc[page_num]
            logger.info(f"\\n=== Debugging Page {page_num + 1} ===")
            
            # Get page dimensions
            page_rect = page.rect
            logger.info(f"Page dimensions: {page_rect.width} x {page_rect.height}")
            
            # Try different text extraction methods
            logger.info("\\n--- Method 1: Simple text extraction ---")
            simple_text: str = page.get_text()
            logger.info(f"Simple text length: {len(simple_text)}")
            if simple_text.strip():
                logger.info(f"First 200 chars: {repr(simple_text[:200])}")
            else:
                logger.warning("No text found with simple extraction")
            
            logger.info("\\n--- Method 2: Dict extraction ---")
            page_dict: Dict[str, Any] = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)
            
            logger.info(f"Page dict keys: {list(page_dict.keys())}")
            logger.info(f"Number of blocks: {len(page_dict.get('blocks', []))}")
            
            text_blocks = 0
            image_blocks = 0
            total_spans = 0
            
            for block_idx, block in enumerate(page_dict.get("blocks", [])):
                logger.info(f"\\nBlock {block_idx}: type={block.get('type', 'unknown')}")
                
                if block["type"] == 0:  # Text block
                    text_blocks += 1
                    logger.info(f"  Text block with {len(block.get('lines', []))} lines")
                    
                    for line_idx, line in enumerate(block.get("lines", [])):
                        logger.info(f"    Line {line_idx}: {len(line.get('spans', []))} spans")
                        
                        for span_idx, span in enumerate(line.get("spans", [])):
                            total_spans += 1
                            text = span.get("text", "")
                            bbox = span.get("bbox", [])
                            font = span.get("font", "")
                            size = span.get("size", 0)
                            
                            if span_idx < 3:  # Show first 3 spans per line
                                logger.info(f"      Span {span_idx}: '{text[:50]}...' bbox={bbox} font={font} size={size}")
                            elif span_idx == 3:
                                remaining = len(line.get("spans", [])) - 3
                                if remaining > 0:
                                    logger.info(f"      ... and {remaining} more spans")
                                
                elif block["type"] == 1:  # Image block
                    image_blocks += 1
                    bbox = block.get("bbox", [])
                    logger.info(f"  Image block: bbox={bbox}")
                    
                else:
                    logger.warning(f"  Unknown block type: {block['type']}")
            
            logger.info(f"\\nPage {page_num + 1} summary:")
            logger.info(f"  Text blocks: {text_blocks}")
            logger.info(f"  Image blocks: {image_blocks}")
            logger.info(f"  Total spans: {total_spans}")
            
            if total_spans == 0:
                logger.warning("❌ No spans found - this explains why no nodes are created!")
                
                # Try alternative extraction methods
                logger.info("\\n--- Trying alternative extraction methods ---")
                
                # Method with different flags
                logger.info("Trying with TEXT_PRESERVE_WHITESPACE flag...")
                alt_dict: Dict[str, Any] = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                alt_blocks = len(alt_dict.get("blocks", []))
                logger.info(f"Alternative method found {alt_blocks} blocks")
                
                # Try raw text blocks
                logger.info("Trying get_text_blocks()...")
                try:
                    text_blocks_list: List[Any] = page.get_text_blocks()
                    logger.info(f"get_text_blocks() returned {len(text_blocks_list)} blocks")
                    for i, block in enumerate(text_blocks_list[:3]):
                        logger.info(f"  Block {i}: {repr(block[4][:100])}")  # text is at index 4
                except Exception as e:
                    logger.error(f"get_text_blocks() failed: {e}")
                
                # Check if page is empty or corrupted
                logger.info("Checking page annotations and drawings...")
                annots = page.annots()
                drawings = page.get_drawings()
                logger.info(f"Annotations: {len(list(annots))}")
                logger.info(f"Drawings: {len(drawings)}")
                
            else:
                logger.info("✅ Found spans - should be able to create nodes")
                break  # We found content, no need to check more pages
        
        doc.close()
        return True
        
    except Exception as e:
        logger.error(f"Error during PDF debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    logger.info("Starting PDF parsing debug...")
    logger.info("=" * 60)
    
    success = debug_pdf_parsing()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ Debug completed!")
    else:
        logger.error("❌ Debug failed!")

if __name__ == "__main__":
    main()

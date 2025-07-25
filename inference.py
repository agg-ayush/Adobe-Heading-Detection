import argparse
import logging
import math
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple, Sequence, MutableSequence, Optional, Generator, Iterable, Any, Mapping

import cv2
import easyocr
import networkx as nx
import numpy as np
import tesserocr
import torch
from shapely import Polygon
from tesserocr import PyTessBaseAPI
from torch_geometric.data import Data

import models
import fitz  # PyMuPDF
from PIL import Image

from core.common import PageEdges, ImageNode, TextNode, get_bytes_per_pixel, PageNodes
from core.models import GLAMGraphNetwork
from glam_classes import CLASSES_MAP

INVALID_UNICODE = chr(0xFFFD)
EasyocrTextResult = namedtuple("EasyocrTextResult", ["bbox", "text", "confidence"])
MuPDFTextTraceChar = namedtuple("MuPDFTextTraceChar", ["unicode", "glyph", "origin", "bbox"])
logger = logging.getLogger(__name__)


def main():
    # Replace hardcoded paths and settings with CLI arguments
    parser = argparse.ArgumentParser(description="Run GLAM inference on a PDF document")
    parser.add_argument("input_pdf", help="Input PDF file path")
    parser.add_argument("model", help="Path to GLAM model file")
    parser.add_argument("--easyocr-langs", nargs="+", default=["en", "ar"], dest="easyocr_languages",
                        help="Languages for EasyOCR reader")
    parser.add_argument("--tesserocr-langs", nargs="+", default=["eng", "ara"], dest="tesserocr_languages",
                        help="Languages for Tesserocr API")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    pdf_filepath = args.input_pdf
    model_filepath = args.model
    easyocr_languages = args.easyocr_languages
    tesserocr_languages = args.tesserocr_languages

    # Configure logging based on verbosity
    logger.handlers.clear()
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # reader = easyocr.Reader(easyocr_languages)
    api = PyTessBaseAPI(path="/usr/share/tesseract/tessdata", lang="+".join(tesserocr_languages))  # type: ignore

    model = GLAMGraphNetwork(PageNodes.features_len, PageEdges.features_len, 512, len(CLASSES_MAP))
    model.load_state_dict(torch.load(model_filepath))
    model = model.to(device)
    model.eval()

    doc = fitz.Document(pdf_filepath)
    for page in doc:
        # Find all nodes
        page_nodes = PageNodes()
        page_dict = fitz.utils.get_text(
            page=page,
            option="dict",
            flags=fitz.TEXT_PRESERVE_IMAGES
        )
        for block in page_dict["blocks"]:  # type: ignore
            if block["type"] == 0:  # type: ignore
                for line in block["lines"]:  # type: ignore
                    for span in line["spans"]:  # type: ignore
                        text = span["text"]  # type: ignore

                        if INVALID_UNICODE in text:
                            ls = " " * (len(text) - len(text.lstrip()))
                            rs = " " * (len(text) - len(text.rstrip()))
                            pixmap = fitz.utils.get_pixmap(
                                page=page,
                                matrix=fitz.Matrix(5, 5),
                                clip=span["bbox"],  # type: ignore
                                colorspace=fitz.csGRAY,
                            )

                            bpp = get_bytes_per_pixel(pixmap.colorspace, pixmap.alpha)
                            api.SetImageBytes(
                                imagedata=pixmap.samples,  # type: ignore
                                width=pixmap.w,
                                height=pixmap.h,
                                bytes_per_pixel=bpp,
                                bytes_per_line=pixmap.stride,
                            )
                            api.SetPageSegMode(tesserocr.PSM.RAW_LINE)
                            api.Recognize()
                            ocr_text = api.GetUTF8Text().rstrip()

                            old_text, text = text, ls + ocr_text + rs
                            logger.debug(f"Replaced {old_text!r} with {text!r}")

                        page_nodes.append(TextNode.from_span(span, text=text))  # type: ignore
            elif block["type"] == 1:  # type: ignore
                page_nodes.append(ImageNode.from_page_block(block))  # type: ignore
            else:
                raise ValueError(f"Unknown block type {block['type']}")  # type: ignore

        # Find all edges
        page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)

        node_features = page_nodes.to_node_features()
        edge_index = page_edges.to_edge_index().t()
        edge_features = page_edges.to_edge_features()
        # print("node_features.shape", node_features.shape, "edge_index.shape", edge_index.shape, "edge_features.shape", edge_features.shape)

        if edge_index.shape[0] == 0:
            continue

        example = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

        with torch.no_grad():
            node_class_scores, edge_class_scores = model(example)
        print("node_class_scores", node_class_scores.shape, "edge_class_scores", edge_class_scores.shape)

        edge_prob_threshold = 0.5
        graph = nx.Graph()
        for k in range(example.edge_index.shape[1]):  # type: ignore
            src_node_i = example.edge_index[0, k].item()  # type: ignore
            dst_node_i = example.edge_index[1, k].item()  # type: ignore
            edge_prob = edge_class_scores[k].item()

            if edge_prob >= edge_prob_threshold:
                graph.add_edge(src_node_i, dst_node_i, weight=edge_prob)
            else:
                graph.add_node(src_node_i)
                graph.add_node(dst_node_i)

        clusters: list[set[int]] = list(nx.connected_components(graph))
        cluster_min_spanning_boxes: list[Polygon] = [
            Polygon([
                (min(page_nodes[node_i].bbox_min_x for node_i in cluster), min(page_nodes[node_i].bbox_min_y for node_i in cluster)),
                (max(page_nodes[node_i].bbox_max_x for node_i in cluster), min(page_nodes[node_i].bbox_min_y for node_i in cluster)),
                (max(page_nodes[node_i].bbox_max_x for node_i in cluster), max(page_nodes[node_i].bbox_max_y for node_i in cluster)),
                (min(page_nodes[node_i].bbox_min_x for node_i in cluster), max(page_nodes[node_i].bbox_max_y for node_i in cluster)),
            ])
            for cluster in clusters
        ]
        cluster_classes: list[int] = torch.stack([node_class_scores[torch.tensor(list(cluster))].sum(dim=0) for cluster in clusters]).argmax(dim=1).tolist()

        print("clusters", clusters)
        print("cluster_min_spanning_boxes", cluster_min_spanning_boxes)
        print("cluster_classes", cluster_classes)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Visual-Focused Window Characteristic Agent
Focuses specifically on diagrams, drawings, and visual content - NOT text regions
"""
import json
import os
import pathlib
import sys
import time
import hashlib
import argparse
import io
import base64
import tempfile
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

# Computer Vision imports
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    CV_AVAILABLE = True
    print("Computer Vision available for visual content detection")
except ImportError:
    print("OpenCV/PIL not available - visual detection disabled")

# PDF to image conversion
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print("PDF2Image available for conversion")
except ImportError:
    print("pdf2image not available")

# Docling for tables only
DOCLING_OK = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import TableItem, TextItem
    DOCLING_OK = True
    print("Docling available for table extraction")
except Exception as e:
    print(f"Docling not available: {e}")

@dataclass
class VisualParameters:
    """Parameters focused on visual content detection"""
    # Visual detection parameters
    min_diagram_size: int = 300  # Minimum size for diagrams
    max_text_ratio: float = 0.3  # Maximum text content ratio to be considered visual
    edge_density_threshold: float = 0.05  # Minimum edge density for diagrams
    
    # Reference matching parameters
    visual_similarity_threshold: float = 0.3  # Lower for visual matching
    sift_min_matches: int = 8  # Minimum SIFT matches required
    
    # Content filtering
    skip_text_heavy_regions: bool = True  # Skip regions with too much text
    require_visual_content: bool = True  # Must contain visual elements
    
    # Processing limits
    max_extractions_per_page: int = 3
    skip_pages: int = 2
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class VisualContentDetector:
    """Specialized detector for visual content like diagrams and drawings"""
    
    def __init__(self):
        if not CV_AVAILABLE:
            raise RuntimeError("OpenCV is required for visual content detection")
    
    def detect_visual_regions(self, image: Image.Image, debug: bool = False) -> List[Dict]:
        """Detect regions that contain visual content (diagrams, drawings) NOT text"""
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        if debug:
            print(f"    Analyzing {width}x{height} image for visual content...")
        
        visual_regions = []
        
        try:
            # Step 1: Detect text regions to EXCLUDE them
            text_regions = self._detect_text_regions(gray, debug)
            
            # Step 2: Detect drawing/diagram regions
            drawing_regions = self._detect_drawing_regions(gray, debug)
            
            # Step 3: Filter out text-heavy regions
            visual_candidates = []
            for region in drawing_regions:
                x, y, w, h = region['bbox']
                
                # Check if this region overlaps significantly with text
                overlaps_text = False
                for text_region in text_regions:
                    tx, ty, tw, th = text_region
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x + w, tx + tw) - max(x, tx))
                    overlap_y = max(0, min(y + h, ty + th) - max(y, ty))
                    overlap_area = overlap_x * overlap_y
                    region_area = w * h
                    
                    if overlap_area / region_area > 0.5:  # 50% overlap with text
                        overlaps_text = True
                        break
                
                if not overlaps_text:
                    visual_candidates.append(region)
            
            # Step 4: Validate visual content
            for region in visual_candidates:
                if self._is_visual_content(cv_image, region, debug):
                    visual_regions.append(region)
            
            if debug:
                print(f"    Found {len(drawing_regions)} drawing regions, {len(visual_candidates)} non-text, {len(visual_regions)} confirmed visual")
        
        except Exception as e:
            if debug:
                print(f"    Visual detection error: {e}")
        
        return visual_regions[:6]  # Limit to 6 per page
    
    def _detect_text_regions(self, gray: np.ndarray, debug: bool = False) -> List[Tuple]:
        """Detect regions that are primarily text"""
        text_regions = []
        
        try:
            # Use morphological operations to find text-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Horizontal kernel for text
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Apply threshold
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = w * h
                
                # Text characteristics: wide, not too tall, reasonable size
                if (aspect_ratio > 3 and  # Wide regions (typical for text lines)
                    h < gray.shape[0] * 0.1 and  # Not too tall
                    area > 500):  # Reasonable size
                    text_regions.append((x, y, w, h))
            
            if debug:
                print(f"      Detected {len(text_regions)} text regions")
        
        except Exception as e:
            if debug:
                print(f"      Text detection error: {e}")
        
        return text_regions
    
    def _detect_drawing_regions(self, gray: np.ndarray, debug: bool = False) -> List[Dict]:
        """Detect regions that might contain drawings or diagrams"""
        drawing_regions = []
        height, width = gray.shape
        
        try:
            # Enhanced edge detection for finding drawn content
            edges = cv2.Canny(gray, 30, 100)
            
            # Use larger kernel to connect drawing elements
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            edges_dilated = cv2.dilate(edges, kernel, iterations=3)
            
            # Find contours of potential drawing regions
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter for drawing-like regions
                if (w >= 200 and h >= 200 and  # Minimum size for meaningful diagrams
                    area >= 40000 and  # Substantial area
                    area <= width * height * 0.8 and  # Not the whole page
                    0.2 <= w/h <= 5):  # Reasonable aspect ratio
                    
                    drawing_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': w/h
                    })
            
            # Sort by area (larger regions first)
            drawing_regions.sort(key=lambda x: x['area'], reverse=True)
            
            if debug:
                print(f"      Detected {len(drawing_regions)} potential drawing regions")
        
        except Exception as e:
            if debug:
                print(f"      Drawing detection error: {e}")
        
        return drawing_regions
    
    def _is_visual_content(self, cv_image: np.ndarray, region: Dict, debug: bool = False) -> bool:
        """Verify that a region contains visual content rather than text"""
        
        x, y, w, h = region['bbox']
        roi = cv_image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        try:
            # 1. Check edge density - visual content has more edges
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            # 2. Check for line structures (common in diagrams)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            
            # 3. Check variance in pixel intensities (diagrams have more variation)
            intensity_variance = np.var(gray_roi)
            
            # 4. Check for geometric shapes (circles, rectangles)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            geometric_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Ignore tiny contours
                    # Check for circular shapes
                    (cx, cy), radius = cv2.minEnclosingCircle(contour)
                    circle_area = np.pi * radius * radius
                    if abs(area - circle_area) / circle_area < 0.3:  # Close to circular
                        geometric_shapes += 1
                    
                    # Check for rectangular shapes
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:  # Rectangle-like
                        geometric_shapes += 1
            
            # Decision criteria for visual content
            is_visual = (
                edge_density > 0.02 and  # Has sufficient edges
                (line_count > 5 or geometric_shapes > 2) and  # Has lines or shapes
                intensity_variance > 500  # Has visual variation
            )
            
            if debug:
                print(f"        Region {region['bbox']}: edges={edge_density:.4f}, lines={line_count}, shapes={geometric_shapes}, var={intensity_variance:.0f} -> {'VISUAL' if is_visual else 'TEXT'}")
            
            return is_visual
        
        except Exception as e:
            if debug:
                print(f"        Visual validation error: {e}")
            return False

class WindowCharacteristicVisualExtractor:
    """Extractor focused on visual window characteristic content"""
    
    def __init__(self, characteristic: str):
        self.characteristic = characteristic
        self.labeled_data_path = pathlib.Path("labeled_data") / characteristic
        
        # Keywords for table filtering only
        self.keywords = {
            'anchors': ['anchor', 'fastener', 'screw', 'bolt', 'connection', 'mounting'],
            'glazing': ['glass', 'glazing', 'IGU', 'thermal', 'coating', 'insulated'],
            'impact_rating': ['impact', 'missile', 'ASTM', 'hurricane', 'rating'],
            'design_pressure': ['pressure', 'DP', 'wind', 'load', 'PSF', 'structural']
        }.get(characteristic, [])
        
        self.params = self._load_parameters()
        self.visual_detector = VisualContentDetector() if CV_AVAILABLE else None
        self.reference_data = self._load_reference_data()
        
        print(f"Initialized {characteristic} visual extractor:")
        print(f"  Reference images: {len(self.reference_data)}")
        print(f"  Visual detection: {'Enabled' if self.visual_detector else 'Disabled'}")
    
    def _load_parameters(self) -> VisualParameters:
        """Load visual parameters"""
        param_file = pathlib.Path(f"visual_params_{self.characteristic}.json")
        if param_file.exists():
            try:
                with open(param_file) as f:
                    data = json.load(f)
                return VisualParameters.from_dict(data)
            except Exception as e:
                print(f"Error loading visual parameters: {e}")
        
        default_params = VisualParameters()
        self._save_parameters(default_params)
        return default_params
    
    def _save_parameters(self, params: VisualParameters = None):
        """Save visual parameters"""
        if params is None:
            params = self.params
        with open(f"visual_params_{self.characteristic}.json", 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def _load_reference_data(self) -> List[Dict]:
        """Load and validate reference visual data"""
        reference_data = []
        
        if not self.labeled_data_path.exists():
            print(f"Reference directory missing: {self.labeled_data_path}")
            print(f"Run: python adaptive_agent.py --setup-reference-data")
            return reference_data
        
        print(f"Loading reference data from: {self.labeled_data_path}")
        
        # Load reference images
        for pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in self.labeled_data_path.glob(pattern):
                try:
                    if CV_AVAILABLE:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            
                            # Validate this is visual content
                            if self._validate_reference_image(img, img_path.name):
                                features = self._extract_visual_features(img)
                                
                                reference_data.append({
                                    'type': 'image',
                                    'path': str(img_path),
                                    'name': img_path.stem,
                                    'data': img,
                                    'features': features,
                                    'size': (w, h)
                                })
                                print(f"  Loaded visual reference: {img_path.name} ({w}x{h})")
                            else:
                                print(f"  Skipped non-visual reference: {img_path.name}")
                        else:
                            print(f"  Could not read: {img_path.name}")
                except Exception as e:
                    print(f"  Error loading {img_path.name}: {e}")
        
        if len(reference_data) == 0:
            print(f"WARNING: No visual reference images found for {self.characteristic}")
            print("Add clear diagram/drawing reference images to improve matching")
        
        return reference_data
    
    def _validate_reference_image(self, img: np.ndarray, name: str) -> bool:
        """Validate that reference image contains visual content"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        
        # Check for lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        line_count = len(lines) if lines is not None else 0
        
        is_visual = edge_density > 0.01 and line_count > 3
        
        if not is_visual:
            print(f"    WARNING: {name} appears to be text/low visual content (edges: {edge_density:.4f}, lines: {line_count})")
        
        return is_visual
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract features specifically for visual content"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        try:
            # SIFT features for visual elements
            sift = cv2.SIFT_create(nfeatures=300)
            kp, desc = sift.detectAndCompute(gray, None)
            features['sift'] = {'keypoints': len(kp), 'descriptors': desc}
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / gray.size
            
            # Line features
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            features['line_count'] = len(lines) if lines is not None else 0
            
            # Contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['contour_count'] = len([c for c in contours if cv2.contourArea(c) > 100])
            
            # Visual histogram
            features['histogram'] = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Reduced bins
            
        except Exception as e:
            print(f"    Feature extraction error: {e}")
        
        return features
    
    def extract_from_document(self, pdf_path: str, pdf_images: List[Image.Image], 
                            docling_result=None, debug: bool = False) -> List[Dict]:
        """Extract visual content and relevant tables"""
        extracted_items = []
        
        print(f"Extracting visual {self.characteristic} content...")
        
        # Extract visual content (diagrams/drawings)
        if self.visual_detector and self.reference_data:
            visual_items = self._extract_visual_content(pdf_images, debug)
            extracted_items.extend(visual_items)
            print(f"  Visual items: {len(visual_items)}")
        else:
            print(f"  Visual extraction skipped (detector: {bool(self.visual_detector)}, references: {len(self.reference_data)})")
        
        # Extract relevant tables
        if docling_result:
            table_items = self._extract_relevant_tables(docling_result, debug)
            extracted_items.extend(table_items)
            print(f"  Table items: {len(table_items)}")
        
        return extracted_items
    
    def _extract_visual_content(self, pdf_images: List[Image.Image], debug: bool = False) -> List[Dict]:
        """Extract visual content from PDF images"""
        visual_items = []
        
        for page_num, page_image in enumerate(pdf_images, 1):
            if page_num <= self.params.skip_pages:
                if debug:
                    print(f"    Skipping page {page_num}")
                continue
            
            if debug:
                print(f"    Processing page {page_num} for visual content...")
            
            # Detect visual regions
            visual_regions = self.visual_detector.detect_visual_regions(page_image, debug)
            
            # Match against reference data
            for region in visual_regions[:self.params.max_extractions_per_page]:
                x, y, w, h = region['bbox']
                region_image = page_image.crop((x, y, w, h))
                
                match_result = self._match_against_references(region_image, region, debug)
                
                if match_result:
                    visual_items.append({
                        'type': f'{self.characteristic}_visual',
                        'content': f"Visual {self.characteristic.replace('_', ' ')} from page {page_num}",
                        'page': page_num,
                        'confidence': match_result['confidence'],
                        'data_uri': self._pil_to_data_uri(region_image),
                        'extraction_id': f"visual_{self.characteristic}_{page_num}_{len(visual_items)}",
                        'bbox': region['bbox'],
                        'metadata': {
                            'extraction_method': 'visual_content_detection',
                            'visual_features': match_result['features'],
                            'reference_match': match_result['reference'],
                            'area': region.get('area', w * h)
                        }
                    })
        
        return visual_items
    
    def _match_against_references(self, region_image: Image.Image, region: Dict, debug: bool = False) -> Optional[Dict]:
        """Match extracted visual region against reference images"""
        
        if not self.reference_data:
            return None
        
        # Convert to OpenCV format
        cv_region = cv2.cvtColor(np.array(region_image), cv2.COLOR_RGB2BGR)
        region_features = self._extract_visual_features(cv_region)
        
        best_match = None
        best_confidence = 0.0
        
        for ref_item in self.reference_data:
            confidence = self._calculate_visual_similarity(region_features, ref_item['features'])
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = ref_item['name']
        
        if debug:
            print(f"      Region {region['bbox']}: best match '{best_match}' (confidence: {best_confidence:.3f})")
        
        # Only accept if meets threshold
        if best_confidence >= self.params.visual_similarity_threshold:
            return {
                'confidence': best_confidence,
                'reference': best_match,
                'features': {
                    'edge_density': region_features.get('edge_density', 0),
                    'line_count': region_features.get('line_count', 0),
                    'contour_count': region_features.get('contour_count', 0)
                }
            }
        
        return None
    
    def _calculate_visual_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between visual features"""
        
        similarity_scores = []
        
        # SIFT feature matching (primary for visual content)
        sift_sim = self._calculate_sift_similarity(features1, features2)
        if sift_sim > 0:
            similarity_scores.append(sift_sim * 0.6)  # 60% weight
        
        # Visual structure similarity
        edge1 = features1.get('edge_density', 0)
        edge2 = features2.get('edge_density', 0)
        if edge1 > 0 and edge2 > 0:
            edge_sim = 1 - abs(edge1 - edge2) / max(edge1, edge2)
            similarity_scores.append(edge_sim * 0.2)  # 20% weight
        
        # Line structure similarity
        line1 = features1.get('line_count', 0)
        line2 = features2.get('line_count', 0)
        if line1 > 0 and line2 > 0:
            line_sim = min(line1, line2) / max(line1, line2)
            similarity_scores.append(line_sim * 0.2)  # 20% weight
        
        return sum(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_sift_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate SIFT similarity for visual matching"""
        try:
            sift1 = features1.get('sift', {})
            sift2 = features2.get('sift', {})
            
            desc1 = sift1.get('descriptors')
            desc2 = sift2.get('descriptors')
            
            if desc1 is None or desc2 is None or len(desc1) < 5 or len(desc2) < 5:
                return 0.0
            
            # Use BF matcher for more reliable results with visual content
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Need minimum matches for visual content
            if len(good_matches) < self.params.sift_min_matches:
                return 0.0
            
            # Calculate similarity
            max_possible = min(len(desc1), len(desc2))
            similarity = len(good_matches) / max_possible
            
            return min(similarity, 1.0)
            
        except Exception:
            return 0.0
    
    def _extract_relevant_tables(self, docling_result, debug: bool = False) -> List[Dict]:
        """Extract tables relevant to the characteristic"""
        relevant_tables = []
        
        if not DOCLING_OK or not docling_result:
            return relevant_tables
        
        try:
            doc = docling_result.document
            all_elements = list(doc.iterate_items())
            table_items = [element for element, _ in all_elements if isinstance(element, TableItem)]
            
            for i, table_element in enumerate(table_items):
                try:
                    table_content = table_element.export_to_markdown()
                    page_num = self._get_page_number(table_element)
                    
                    if len(table_content.strip()) < 100:
                        continue
                    
                    # Simple keyword-based relevance
                    relevance_score = sum(1 for keyword in self.keywords if keyword.lower() in table_content.lower())
                    
                    if debug:
                        print(f"    Table {i+1} (page {page_num}): {relevance_score} keyword matches")
                    
                    if relevance_score >= 2:  # At least 2 keyword matches
                        relevant_tables.append({
                            'type': f'{self.characteristic}_table',
                            'content': table_content,
                            'page': page_num,
                            'confidence': min(relevance_score / 5.0, 1.0),
                            'extraction_id': f"table_{self.characteristic}_{page_num}_{i}",
                            'metadata': {
                                'extraction_method': 'keyword_table_filtering',
                                'keyword_matches': relevance_score
                            }
                        })
                        
                        if len(relevant_tables) >= 5:  # Limit tables
                            break
                
                except Exception as e:
                    if debug:
                        print(f"    Table {i+1}: Error - {e}")
        
        except Exception as e:
            if debug:
                print(f"  Table extraction error: {e}")
        
        return relevant_tables
    
    def _get_page_number(self, element) -> int:
        """Get page number from element"""
        try:
            if hasattr(element, 'prov') and element.prov and len(element.prov) > 0:
                return int(element.prov[0].page_no)
        except:
            pass
        return 1
    
    def _pil_to_data_uri(self, pil_image) -> str:
        """Convert PIL image to data URI"""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

class VisualWindowDocumentProcessor:
    """Document processor focused on visual content extraction"""
    
    def __init__(self):
        print("Visual Window Document Processor - Focus on diagrams and drawings")
        
        self.characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
        
        # Initialize visual extractors
        self.extractors = {}
        for characteristic in self.characteristics:
            try:
                self.extractors[characteristic] = WindowCharacteristicVisualExtractor(characteristic)
            except Exception as e:
                print(f"Failed to initialize {characteristic} extractor: {e}")
        
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is required")
        
        # Setup Docling for tables
        if DOCLING_OK:
            pp = PdfPipelineOptions()
            pp.generate_table_images = True
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pp)}
            )
            print("Docling configured for table extraction")
        
        self._check_azure_config()
    
    def _check_azure_config(self):
        """Check Azure OpenAI configuration"""
        try:
            load_dotenv(override=True)
        except:
            pass
        
        required = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required if not os.getenv(var)]
        
        if missing:
            print(f"Azure OpenAI missing: {missing}")
        else:
            print("Azure OpenAI configured for LLM feedback")
    
    def process_document(self, source: str, characteristics: List[str] = None, debug: bool = False) -> Dict[str, str]:
        """Process document focusing on visual content"""
        start_time = time.time()
        base_doc_id = hashlib.md5(f"{source}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        if characteristics is None:
            characteristics = [c for c in self.characteristics if c in self.extractors]
        
        print(f"\nVISUAL WINDOW DOCUMENT PROCESSING")
        print(f"Source: {os.path.basename(source)}")
        print(f"Document ID: {base_doc_id}")
        print(f"Focus: Visual content (diagrams/drawings) + relevant tables")
        print(f"Characteristics: {', '.join(characteristics)}")
        
        # Download if URL
        if source.startswith(("http://", "https://")):
            source = self._download(source)
        
        # Convert PDF to images
        print(f"\nSTEP 1: PDF CONVERSION")
        pdf_images = self._convert_pdf_to_images(source, debug)
        print(f"Converted {len(pdf_images)} pages")
        
        # Extract tables
        print(f"\nSTEP 2: TABLE EXTRACTION")
        docling_result = None
        if DOCLING_OK:
            try:
                docling_result = self.converter.convert(source)
                print(f"Docling processing completed")
            except Exception as e:
                print(f"Docling failed: {e}")
        
        # Process each characteristic
        print(f"\nSTEP 3: VISUAL CONTENT EXTRACTION")
        extraction_results = {}
        
        for characteristic in characteristics:
            if characteristic not in self.extractors:
                continue
            
            print(f"\nProcessing {characteristic.upper()}...")
            
            extractor = self.extractors[characteristic]
            extracted_items = extractor.extract_from_document(
                source, pdf_images, docling_result, debug
            )
            
            # Save results
            extraction_data = {
                'document_id': base_doc_id,
                'characteristic': characteristic,
                'document_path': source,
                'document_type': 'visual_window_document',
                'extracted_sections': extracted_items,
                'text_summary': {},  # Not used in visual approach
                'enhancement_features': {
                    'visual_content_detection': True,
                    'text_region_filtering': True,
                    'reference_visual_matching': len(extractor.reference_data) > 0
                },
                'parameters_used': extractor.params.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'total_sections': len(extracted_items),
                'total_pages': len(pdf_images),
                'extraction_summary': {
                    'visual_items': len([i for i in extracted_items if 'visual' in i.get('type', '')]),
                    'table_items': len([i for i in extracted_items if 'table' in i.get('type', '')]),
                    'avg_confidence': sum(item.get('confidence', 0) for item in extracted_items) / len(extracted_items) if extracted_items else 0
                },
                'processing_method': 'visual_content_focused_extraction'
            }
            
            pathlib.Path("feedback_data").mkdir(exist_ok=True)
            extraction_file = f"feedback_data/{characteristic}_extraction_{base_doc_id}.json"
            with open(extraction_file, 'w') as f:
                json.dump(extraction_data, f, indent=2)
            
            print(f"Extracted {len(extracted_items)} items")
            print(f"Saved: {extraction_file}")
            
            extraction_results[characteristic] = base_doc_id
        
        # Run LLM feedback
        print(f"\nSTEP 4: VISUAL-AWARE LLM FEEDBACK")
        for characteristic in characteristics:
            if characteristic in extraction_results:
                print(f"\nRunning {characteristic.upper()} visual feedback...")
                success = self._run_visual_llm_feedback(
                    characteristic, extraction_results[characteristic], source, debug
                )
                if success:
                    print(f"{characteristic.title()} feedback completed")
                else:
                    print(f"{characteristic.title()} used fallback")
        
        processing_time = time.time() - start_time
        
        print(f"\nVISUAL PROCESSING COMPLETED")
        print(f"Processing time: {processing_time:.1f}s")
        for char, doc_id in extraction_results.items():
            try:
                with open(f"feedback_data/{char}_extraction_{doc_id}.json") as f:
                    data = json.load(f)
                    items = len(data.get('extracted_sections', []))
                    visual_items = len([i for i in data.get('extracted_sections', []) if 'visual' in i.get('type', '')])
                    print(f"  {char.title()}: {items} total ({visual_items} visual)")
            except:
                pass
        
        return extraction_results
    
    def _convert_pdf_to_images(self, pdf_path: str, debug: bool = False) -> List[Image.Image]:
        """Convert PDF to high-quality images for visual analysis"""
        try:
            # Higher DPI for better visual content detection
            images = convert_from_path(pdf_path, dpi=300, fmt='RGB')
            if debug:
                print(f"  Converted at 300 DPI for visual analysis")
            return images
        except Exception as e:
            print(f"PDF conversion failed: {e}")
            raise
    
    def _run_visual_llm_feedback(self, characteristic: str, doc_id: str, source_pdf: str, debug: bool = False) -> bool:
        """Run LLM feedback with visual awareness"""
        try:
            required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                return self._run_visual_fallback(characteristic, doc_id)
            
            cmd = [
                sys.executable, "llm_feedback.py", 
                "--enhanced-analyze", characteristic, doc_id,
                "--source-pdf", source_pdf
            ]
            if debug:
                cmd.append("--debug")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
            
        except Exception:
            return self._run_visual_fallback(characteristic, doc_id)
    
    def _run_visual_fallback(self, characteristic: str, doc_id: str) -> bool:
        """Fallback analysis for visual content"""
        try:
            extraction_file = f"feedback_data/{characteristic}_extraction_{doc_id}.json"
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            sections = extraction_data.get('extracted_sections', [])
            visual_items = [s for s in sections if 'visual' in s.get('type', '')]
            table_items = [s for s in sections if 'table' in s.get('type', '')]
            
            print(f"Visual fallback: {len(visual_items)} visual, {len(table_items)} tables")
            
            # Simple adjustments based on visual content
            adjustments = {}
            
            if len(visual_items) == 0:
                adjustments['visual_similarity_threshold'] = 0.2  # Lower threshold
                print("  No visual content found - lowering similarity threshold")
            
            if len(visual_items) > 8:
                adjustments['visual_similarity_threshold'] = 0.4  # Higher threshold
                print("  Too much visual content - raising similarity threshold")
            
            # Apply adjustments
            if adjustments:
                param_file = f"visual_params_{characteristic}.json"
                try:
                    if os.path.exists(param_file):
                        with open(param_file) as f:
                            current_params = json.load(f)
                    else:
                        current_params = VisualParameters().to_dict()
                    
                    current_params.update(adjustments)
                    
                    with open(param_file, 'w') as f:
                        json.dump(current_params, f, indent=2)
                    
                    print(f"  Applied {len(adjustments)} parameter changes")
                except Exception as e:
                    print(f"  Parameter update failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"Visual fallback failed: {e}")
            return False
    
    def _download(self, url: str) -> str:
        """Download PDF from URL"""
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return tmp

def setup_visual_reference_data():
    """Setup reference data structure optimized for visual content"""
    characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
    labeled_data_path = pathlib.Path("labeled_data")
    
    for characteristic in characteristics:
        char_path = labeled_data_path / characteristic
        char_path.mkdir(parents=True, exist_ok=True)
        
        readme_path = char_path / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"""VISUAL REFERENCE DATA: {characteristic.replace('_', ' ').title()}

CRITICAL: This system focuses on VISUAL CONTENT (diagrams, drawings, technical illustrations)
NOT text paragraphs or written descriptions.

WHAT TO ADD:
- Clear diagrams showing {characteristic.replace('_', ' ')} components
- Technical drawings with lines, shapes, and visual elements
- Cross-section views and detail drawings  
- Assembly diagrams and exploded views
- Photos of actual {characteristic.replace('_', ' ')} hardware/components

AVOID:
- Text-heavy images or screenshots of paragraphs
- Images that are mostly words/descriptions
- Low-contrast or blurry images
- Screenshots of software interfaces

REQUIREMENTS:
- Minimum 300x300 pixels
- Clear visual content with lines, shapes, diagrams
- High contrast between elements
- Multiple examples showing different views/styles

EXAMPLE GOOD REFERENCES FOR {characteristic.upper()}:
""")
            
            if characteristic == 'anchors':
                f.write("""- Cross-section showing screw going into concrete
- Diagram of wood buck attachment detail
- Technical drawing of fastener assembly
- Photo of actual anchor hardware
""")
            elif characteristic == 'glazing':
                f.write("""- IGU cross-section diagram showing glass layers
- Glazing assembly detail drawing
- Thermal break illustration
- Glass specification diagram with dimensions
""")
            elif characteristic == 'impact_rating':
                f.write("""- Impact test result charts/graphs
- Certification mark examples
- Performance rating diagrams
- Test setup illustrations
""")
            elif characteristic == 'design_pressure':
                f.write("""- Pressure load diagrams
- Wind pressure distribution charts
- Structural load illustrations  
- Performance curve graphs
""")
        
        print(f"Created visual reference guide: {char_path}/README.txt")
    
    print("\nVISUAL REFERENCE DATA SETUP COMPLETE")
    print("Key points:")
    print("1. Add DIAGRAMS and DRAWINGS, not text images")
    print("2. Ensure high visual contrast and clear lines/shapes")
    print("3. Use multiple examples per characteristic")
    print("4. Minimum 300x300 pixel resolution")

def main():
    parser = argparse.ArgumentParser(description="Visual-Focused Window Characteristic Agent")
    parser.add_argument("--source", help="PDF path or URL to process")
    parser.add_argument("--characteristics", nargs="+", 
                       choices=['anchors', 'glazing', 'impact_rating', 'design_pressure'],
                       help="Characteristics to extract")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--setup-visual-data", action="store_true", help="Setup visual reference data")
    
    args = parser.parse_args()
    
    if args.setup_visual_data:
        setup_visual_reference_data()
        return 0
    
    if not args.source:
        parser.error("--source is required")
    
    print("VISUAL-FOCUSED WINDOW PROCESSING")
    print("Focus: Diagrams, drawings, and technical illustrations")
    
    try:
        processor = VisualWindowDocumentProcessor()
        characteristics = args.characteristics or processor.characteristics
        
        results = processor.process_document(args.source, characteristics, debug=args.debug)
        
        print("\nSUCCESS - Visual content extracted")
        print("View results: streamlit run feedback_interface.py")
        
        print("\nIMPORTANT: If results show text regions instead of diagrams:")
        print("1. Check your reference images - they must be visual diagrams")
        print("2. Add more diverse visual reference examples")
        print("3. Ensure reference images have clear lines/shapes")
        
        return 0
        
    except Exception as e:
        print(f"FAILED: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
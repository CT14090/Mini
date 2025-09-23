#!/usr/bin/env python3
# characteristic_based_extractor.py
"""
Enhanced Characteristic-Based Content Extractor with Sliding Window Detection
Focuses on finding actual visual content like diagrams, tables, and technical drawings
"""

import json
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple
import base64
import io

# Computer Vision imports
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    CV_AVAILABLE = True
except ImportError:
    pass

class CharacteristicBasedExtractor:
    """Enhanced extractor with sliding window detection for comprehensive coverage"""
    
    def __init__(self):
        self.labeled_data_path = pathlib.Path("labeled_data")
        self.characteristics = self._load_characteristics()
        self.training_data = self._load_training_data()
        
        # Processing limits - restrictive to focus on specific regions
        self.max_regions_per_page = 6
        self.min_region_size = 20000   # At least ~140x140 pixels
        self.max_region_size = 2000000 # No more than ~1400x1400 pixels
        self.processing_timeout_per_page = 20
        
        print(f"✓ Enhanced extractor initialized with {len(self.characteristics)} characteristics")
        print(f"  Region size limits: {self.min_region_size:,} - {self.max_region_size:,} pixels")
    
    def _load_characteristics(self) -> Dict[str, Dict]:
        """Load available characteristics with enhanced metadata"""
        characteristics = {
            'anchors': {
                'name': 'Anchors',
                'description': 'Anchor details, attachment methods, and fastening systems',
                'keywords': ['anchor', 'fastener', 'attachment', 'bolt', 'screw', 'connection'],
                'diagram_keywords': ['detail', 'section', 'assembly', 'connection'],
                'visual_features': ['technical_drawing', 'cross_section', 'detail_callout']
            },
            'design_pressure': {
                'name': 'Design Pressure', 
                'description': 'Design pressure ratings, wind load data, and structural calculations',
                'keywords': ['pressure', 'wind', 'load', 'psf', 'kpa', 'rating'],
                'diagram_keywords': ['chart', 'table', 'rating', 'performance', 'graph'],
                'visual_features': ['table', 'chart', 'graph', 'performance_data']
            },
            'glazing': {
                'name': 'Glazing',
                'description': 'Glass specifications, glazing details, and glazing systems',
                'keywords': ['glass', 'glazing', 'IG', 'IGU', 'pane', 'glazed'],
                'diagram_keywords': ['section', 'detail', 'specification', 'assembly'],
                'visual_features': ['technical_drawing', 'cross_section', 'assembly_detail']
            },
            'impact_rating': {
                'name': 'Impact Rating',
                'description': 'Impact resistance ratings, test results, and compliance data',
                'keywords': ['impact', 'missile', 'test', 'rating', 'zone', 'compliance'],
                'diagram_keywords': ['chart', 'table', 'certification', 'compliance', 'test'],
                'visual_features': ['table', 'certification', 'test_result', 'compliance_chart']
            }
        }
        return characteristics
    
    def _load_training_data(self) -> Dict[str, List[Dict]]:
        """Load training data with enhanced feature extraction"""
        training_data = {}
        
        if not self.labeled_data_path.exists():
            return training_data
        
        for category_dir in self.labeled_data_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            if category_name not in self.characteristics:
                continue
            
            image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            
            training_data[category_name] = []
            
            for img_file in image_files[:8]:  # Increased to 8 per category
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        # Extract visual features from training image
                        features = self._extract_visual_features(img)
                        
                        training_data[category_name].append({
                            'image': img,
                            'path': str(img_file),
                            'filename': img_file.name,
                            'features': features
                        })
                except Exception as e:
                    print(f"  Warning: Error loading {img_file}: {e}")
        
        return training_data
    
    def get_available_characteristics(self) -> List[str]:
        """Get list of available characteristics"""
        return list(self.characteristics.keys())
    
    def get_characteristic_info(self, characteristic: str) -> Dict:
        """Get information about a characteristic"""
        return self.characteristics.get(characteristic, {})
    
    def extract_characteristic_content(self, page_image: Image.Image, characteristic: str, 
                                     page_num: int, debug: bool = False) -> List[Dict]:
        """Extract content for specific characteristic using sliding window detection"""
        if not CV_AVAILABLE:
            return []
        
        if characteristic not in self.characteristics:
            if debug:
                print(f"    Unknown characteristic: {characteristic}")
            return []
        
        start_time = time.time()
        extracted_content = []
        
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            
            # Extract regions using sliding window + traditional methods
            regions = self._detect_visual_content_regions(cv_image, page_image, page_num, debug)
            
            if debug:
                print(f"    Found {len(regions)} visual content regions")
            
            # Classify each region
            for i, region_data in enumerate(regions):
                # Check timeout
                if time.time() - start_time > self.processing_timeout_per_page:
                    if debug:
                        print(f"    Page timeout - processed {i}/{len(regions)} regions")
                    break
                
                classification = self._classify_visual_region(
                    region_data, characteristic, debug
                )
                
                if classification:
                    extracted_content.append(classification)
                    if debug:
                        print(f"      ✓ Classified: {classification['type']} (confidence: {classification['confidence']:.3f})")
                
                # Hard limit on extractions per page
                if len(extracted_content) >= self.max_regions_per_page:
                    if debug:
                        print(f"    Reached max extractions per page ({self.max_regions_per_page})")
                    break
            
            return extracted_content
            
        except Exception as e:
            if debug:
                print(f"    Error extracting content: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def _detect_visual_content_regions(self, cv_image, pil_image: Image.Image, page_num: int, debug: bool = False) -> List[Dict]:
        """Detect regions containing visual content using sliding window + traditional methods"""
        regions = []
        height, width = cv_image.shape[:2]
        
        if debug:
            print(f"    Page dimensions: {width}x{height} pixels")
        
        # Method 1: Traditional detection methods (for comparison)
        table_regions = self._detect_table_regions(cv_image, pil_image, page_num)
        diagram_regions = self._detect_diagram_regions(cv_image, pil_image, page_num)
        content_blocks = self._detect_content_blocks(cv_image, pil_image, page_num)
        
        traditional_regions = table_regions + diagram_regions + content_blocks
        regions.extend(traditional_regions)
        
        if debug:
            print(f"    Traditional detection: {len(table_regions)} tables, {len(diagram_regions)} diagrams, {len(content_blocks)} content blocks")
        
        # Method 2: Sliding window detection at multiple scales
        sliding_regions = self._sliding_window_detection(cv_image, pil_image, page_num, debug)
        regions.extend(sliding_regions)
        
        if debug:
            print(f"    Sliding window detection: {len(sliding_regions)} regions")
        
        # Remove overlapping regions and apply filtering
        filtered_regions = self._filter_and_deduplicate_regions(regions, debug)
        
        if debug:
            print(f"    Final regions after filtering: {len(filtered_regions)}")
        
        return filtered_regions
    
    def _sliding_window_detection(self, cv_image, pil_image: Image.Image, page_num: int, debug: bool = False) -> List[Dict]:
        """Systematic sliding window detection at multiple scales"""
        regions = []
        height, width = cv_image.shape[:2]
        
        # Define multiple window sizes based on typical diagram sizes
        window_sizes = [
            (150, 150),    # Small details, fastener close-ups
            (200, 300),    # Medium diagrams, connection details  
            (300, 200),    # Wide diagrams, horizontal assemblies
            (250, 400),    # Tall diagrams, vertical sections
            (400, 300),    # Large diagrams, complete assemblies
            (300, 500),    # Very tall sections
            (500, 250),    # Very wide assemblies
        ]
        
        total_candidates = 0
        processed_candidates = 0
        
        for window_w, window_h in window_sizes:
            # Skip windows that are too large for the page
            if window_w > width * 0.8 or window_h > height * 0.8:
                continue
                
            step_w = window_w // 2  # 50% overlap
            step_h = window_h // 2
            
            # Scan across the page
            for y in range(0, height - window_h + 1, step_h):
                for x in range(0, width - window_w + 1, step_w):
                    total_candidates += 1
                    
                    # Extract window region
                    window_area = window_w * window_h
                    
                    # Skip if outside our size constraints
                    if not (self.min_region_size <= window_area <= self.max_region_size):
                        continue
                    
                    # Analyze this window
                    roi_cv = cv_image[y:y+window_h, x:x+window_w]
                    
                    # Quick quality check - skip mostly empty or mostly filled regions
                    if not self._window_quality_check(roi_cv):
                        continue
                    
                    processed_candidates += 1
                    
                    # Check if this window contains interesting content
                    content_score = self._evaluate_window_content(roi_cv, debug=False)
                    
                    # Only keep windows with reasonable content scores
                    if content_score > 0.15:  # Threshold for interesting content
                        roi_pil = pil_image.crop((x, y, x + window_w, y + window_h))
                        
                        regions.append({
                            'image': roi_pil,
                            'cv_image': roi_cv,
                            'bbox': (x, y, x + window_w, y + window_h),
                            'page': page_num,
                            'area': window_area,
                            'detection_method': 'sliding_window',
                            'content_type': 'potential_diagram',
                            'content_score': content_score,
                            'window_size': f"{window_w}x{window_h}"
                        })
        
        if debug:
            print(f"      Sliding window stats: {total_candidates} total positions, {processed_candidates} analyzed, {len(regions)} candidates found")
        
        # Sort by content score and keep top candidates
        regions.sort(key=lambda r: r.get('content_score', 0), reverse=True)
        
        # Limit number of sliding window regions to prevent overwhelm
        max_sliding_regions = 12
        regions = regions[:max_sliding_regions]
        
        return regions
    
    def _window_quality_check(self, roi_cv) -> bool:
        """Quick check to skip obviously empty or filled windows"""
        try:
            gray = cv2.cvtColor(roi_cv, cv2.COLOR_BGR2GRAY) if len(roi_cv.shape) == 3 else roi_cv
            
            # Check for reasonable content distribution
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Skip if too uniform (empty or solid color)
            if std_intensity < 15:  # Very low variation
                return False
            
            # Skip if too extreme in brightness
            if mean_intensity < 20 or mean_intensity > 235:  # Too dark or too light
                return False
            
            # Check for some edge content
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Must have some edges but not be overwhelmed with them
            if edge_density < 0.005 or edge_density > 0.4:
                return False
            
            return True
            
        except:
            return False
    
    def _evaluate_window_content(self, roi_cv, debug: bool = False) -> float:
        """Evaluate how likely a window is to contain anchor-related content"""
        try:
            gray = cv2.cvtColor(roi_cv, cv2.COLOR_BGR2GRAY) if len(roi_cv.shape) == 3 else roi_cv
            h, w = gray.shape
            
            content_indicators = []
            
            # 1. Edge structure analysis
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Good technical diagrams have moderate edge density
            if 0.02 < edge_density < 0.2:
                content_indicators.append(0.3)
            elif 0.01 < edge_density < 0.3:
                content_indicators.append(0.15)
            
            # 2. Line structure detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(w//3, 20), 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(h//3, 20)))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_density = np.sum(horizontal_lines > 0) / edges.size
            v_line_density = np.sum(vertical_lines > 0) / edges.size
            
            # Technical drawings often have structured lines
            if h_line_density > 0.005 or v_line_density > 0.005:
                content_indicators.append(0.25)
            
            # 3. Shape complexity
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            # Good diagrams have multiple shapes but not too many
            if 2 <= len(significant_contours) <= 15:
                content_indicators.append(0.2)
            elif 1 <= len(significant_contours) <= 25:
                content_indicators.append(0.1)
            
            # 4. Content distribution
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary > 127) / binary.size
            
            # Technical drawings typically have good white space balance
            if 0.4 < white_ratio < 0.85:
                content_indicators.append(0.15)
            
            # 5. Construction-specific patterns
            # Look for circular/curved elements (bolts, connections)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, 
                                     minDist=max(w, h)//8,
                                     param1=50, param2=30, 
                                     minRadius=3, maxRadius=min(w, h)//6)
            
            if circles is not None and len(circles[0]) > 0:
                # Circles might indicate bolts/fasteners
                content_indicators.append(0.2)
            
            # 6. Text content check (should be minimal for diagrams)
            text_score = self._estimate_text_content(gray)
            if text_score < 0.3:  # Low text content is good for diagrams
                content_indicators.append(0.1)
            
            # Calculate final score
            final_score = sum(content_indicators)
            
            if debug and final_score > 0.2:
                print(f"        Window content score: {final_score:.3f} (indicators: {len(content_indicators)})")
            
            return min(1.0, final_score)
            
        except Exception as e:
            if debug:
                print(f"        Window evaluation error: {e}")
            return 0.0
    
    def _estimate_text_content(self, gray) -> float:
        """Estimate how much text vs diagram content is in the region"""
        try:
            # Look for text-like patterns
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_indicators = 0
            total_components = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:  # Skip tiny components
                    continue
                    
                total_components += 1
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text characters have typical size and aspect ratio ranges
                if 0.3 < aspect_ratio < 4 and 8 < w < 50 and 8 < h < 30:
                    text_indicators += 1
            
            return text_indicators / max(1, total_components)
            
        except:
            return 0.0
    
    def _detect_table_regions(self, cv_image, pil_image, page_num) -> List[Dict]:
        """Detect table-like structures"""
        regions = []
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines to find table structures
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        table_structure = cv2.threshold(table_structure, 30, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours in table structure
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Size filtering
            if self.min_region_size <= area <= self.max_region_size and w > 80 and h > 60:
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 8:
                    # Verify table structure
                    roi_table = table_structure[y:y+h, x:x+w]
                    table_density = np.sum(roi_table > 0) / (w * h)
                    
                    if table_density > 0.02:
                        # Add padding
                        pad = 10
                        x_pad = max(0, x - pad)
                        y_pad = max(0, y - pad)
                        w_pad = min(cv_image.shape[1] - x_pad, w + 2*pad)
                        h_pad = min(cv_image.shape[0] - y_pad, h + 2*pad)
                        
                        region_img = pil_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                        
                        regions.append({
                            'image': region_img,
                            'cv_image': cv_image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad],
                            'bbox': (x_pad, y_pad, x_pad + w_pad, y_pad + h_pad),
                            'page': page_num,
                            'area': w_pad * h_pad,
                            'detection_method': 'table_detection',
                            'content_type': 'table',
                            'confidence_boost': 0.2,
                            'table_density': table_density
                        })
        
        return regions
    
    def _detect_diagram_regions(self, cv_image, pil_image, page_num) -> List[Dict]:
        """Detect diagram regions"""
        regions = []
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 40, 120)
        
        # Connect nearby components
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Size filtering
            if self.min_region_size <= area <= self.max_region_size:
                # Edge density analysis
                roi_edges = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi_edges > 0) / roi_edges.size
                
                if 0.01 < edge_density < 0.2:
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    if self._has_diagram_structure(roi_gray):
                        aspect_ratio = w / h
                        if 0.2 < aspect_ratio < 5:
                            # Add padding
                            pad = 5
                            x_pad = max(0, x - pad)
                            y_pad = max(0, y - pad)
                            w_pad = min(cv_image.shape[1] - x_pad, w + 2*pad)
                            h_pad = min(cv_image.shape[0] - y_pad, h + 2*pad)
                            
                            region_img = pil_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                            
                            regions.append({
                                'image': region_img,
                                'cv_image': cv_image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad],
                                'bbox': (x_pad, y_pad, x_pad + w_pad, y_pad + h_pad),
                                'page': page_num,
                                'area': w_pad * h_pad,
                                'detection_method': 'diagram_detection',
                                'content_type': 'diagram',
                                'edge_density': edge_density
                            })
        
        return regions
    
    def _detect_content_blocks(self, cv_image, pil_image, page_num) -> List[Dict]:
        """Detect content blocks"""
        regions = []
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold and morphological operations
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_inv = cv2.bitwise_not(binary)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Size and aspect ratio filtering
            if self.min_region_size <= area <= self.max_region_size:
                aspect_ratio = w / h
                
                if 0.3 < aspect_ratio < 5:
                    # Content density check
                    roi_binary = binary_inv[y:y+h, x:x+w]
                    content_density = np.sum(roi_binary > 0) / roi_binary.size
                    
                    if 0.1 < content_density < 0.7:
                        # Check if this looks like visual content rather than text
                        if self._looks_like_visual_content(gray[y:y+h, x:x+w]):
                            region_img = pil_image.crop((x, y, x + w, y + h))
                            
                            regions.append({
                                'image': region_img,
                                'cv_image': cv_image[y:y+h, x:x+w],
                                'bbox': (x, y, x + w, y + h),
                                'page': page_num,
                                'area': area,
                                'detection_method': 'content_block',
                                'content_type': 'visual_content',
                                'content_density': content_density
                            })
        
        return regions
    
    def _has_diagram_structure(self, roi_gray) -> bool:
        """Check if region has structured diagram-like content"""
        # Look for lines, shapes, and structured elements
        edges = cv2.Canny(roi_gray, 30, 100)
        
        # Check for horizontal/vertical line components
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        h_lines = np.sum(horizontal > 0)
        v_lines = np.sum(vertical > 0)
        total_edges = np.sum(edges > 0)
        
        # Diagrams typically have some structured lines
        if total_edges > 0:
            line_ratio = (h_lines + v_lines) / total_edges
            return line_ratio > 0.1  # At least 10% structured lines
        
        return False
    
    def _looks_like_visual_content(self, roi_gray) -> bool:
        """Enhanced check if region looks like visual content vs pure text"""
        try:
            height, width = roi_gray.shape
            
            # Skip very small regions
            if height < 50 or width < 50:
                return False
            
            # Edge analysis
            edges = cv2.Canny(roi_gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Look for line structures (indicating diagrams/tables)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(width//4, 20), 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(height//4, 20)))
            
            horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_lines = np.sum(horizontal > 0)
            v_lines = np.sum(vertical > 0)
            total_edges = np.sum(edges > 0)
            
            # Check white space distribution (text has more regular patterns)
            white_ratio = np.sum(roi_gray > 200) / (height * width)
            
            # Visual content typically has:
            # - Some edge density (but not too much - that's usually dense text)
            # - Some line structures (tables/diagrams have lines)
            # - Not too much white space (pure text documents are mostly white)
            has_edges = 0.015 < edge_density < 0.15
            has_structure = (h_lines + v_lines) > edge_density * roi_gray.size * 0.1
            reasonable_white = 0.3 < white_ratio < 0.85
            
            return has_edges and (has_structure or reasonable_white)
            
        except Exception:
            return True  # If analysis fails, assume it might be visual content
    
    def _filter_and_deduplicate_regions(self, regions: List[Dict], debug: bool = False) -> List[Dict]:
        """Remove overlapping regions and apply enhanced filtering with ranking"""
        if not regions:
            return []
        
        # First, assign quality scores to all regions
        for region in regions:
            region['quality_score'] = self._calculate_region_quality_score(region)
        
        # Sort by quality score (highest first)
        regions.sort(key=lambda r: r['quality_score'], reverse=True)
        
        if debug:
            print(f"      Region quality scores (top 10):")
            for i, region in enumerate(regions[:10]):
                method = region.get('detection_method', 'unknown')
                score = region.get('quality_score', 0)
                area = region.get('area', 0)
                print(f"        {i+1}. {method}: {score:.3f} ({area:,} pixels)")
        
        filtered_regions = []
        page_area = None
        
        for region in regions:
            bbox = region['bbox']
            x1, y1, x2, y2 = bbox
            area = region['area']
            
            # Calculate page area from first region if not set
            if page_area is None:
                if area > 10000000:
                    page_area = area * 1.2
                else:
                    page_area = 15000000
            
            # Skip regions that are too large (likely full page captures)
            if area > page_area * 0.6:  # More than 60% of page
                if debug:
                    print(f"      Skipping large region: {area:,} pixels (likely full page)")
                continue
            
            # Skip regions that are too small for meaningful content
            if area < self.min_region_size:
                if debug:
                    print(f"      Skipping small region: {area:,} pixels")
                continue
            
            # Check aspect ratio
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            if aspect_ratio > 8 or aspect_ratio < 0.125:
                if debug:
                    print(f"      Skipping extreme aspect ratio: {aspect_ratio:.2f}")
                continue
            
            # Enhanced overlap checking - keep higher quality regions
            overlaps = False
            for existing in filtered_regions:
                ex1, ey1, ex2, ey2 = existing['bbox']
                
                # Calculate overlap
                overlap_area = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                
                # If significant overlap, keep the higher quality one
                overlap_ratio = overlap_area / min(area, existing['area'])
                if overlap_ratio > 0.5:  # 50% overlap threshold
                    # Current region has higher quality score, so remove existing
                    if region['quality_score'] > existing['quality_score']:
                        filtered_regions.remove(existing)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                filtered_regions.append(region)
                
                if debug:
                    method = region.get('detection_method', 'unknown')
                    score = region.get('quality_score', 0)
                    print(f"      Added region: {area:,} pixels, aspect: {aspect_ratio:.2f}, quality: {score:.3f} ({method})")
                
                # Limit total regions but be more generous for sliding window
                max_regions = self.max_regions_per_page + 6  # Allow more candidates
                if len(filtered_regions) >= max_regions:
                    break
        
        # Final sort by quality score
        filtered_regions.sort(key=lambda r: r['quality_score'], reverse=True)
        
        return filtered_regions
    
    def _calculate_region_quality_score(self, region: Dict) -> float:
        """Calculate a quality score for region ranking"""
        score = 0.0
        
        try:
            # Base score from detection method
            method = region.get('detection_method', 'unknown')
            if method == 'table_detection':
                score += 0.3  # Tables are often good
            elif method == 'diagram_detection':
                score += 0.25  # Diagrams are good
            elif method == 'sliding_window':
                score += 0.2  # Sliding window gets moderate base score
            elif method == 'content_block':
                score += 0.15  # Content blocks are least reliable
            
            # Content score boost
            content_score = region.get('content_score', 0)
            score += content_score * 0.3  # Up to 0.3 boost
            
            # Size scoring - prefer medium-sized regions
            area = region.get('area', 0)
            if 50000 <= area <= 500000:  # Sweet spot for anchor diagrams
                score += 0.2
            elif 25000 <= area <= 1000000:  # Acceptable range
                score += 0.1
            elif area < 25000:  # Too small
                score -= 0.1
            
            # Aspect ratio scoring
            bbox = region.get('bbox', (0, 0, 0, 0))
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                if height > 0:
                    aspect_ratio = width / height
                    if 0.5 <= aspect_ratio <= 3:  # Good aspect ratios
                        score += 0.15
                    elif 0.25 <= aspect_ratio <= 5:  # Acceptable
                        score += 0.05
            
            # Specific feature bonuses
            if 'table_density' in region and region['table_density'] > 0.05:
                score += 0.1  # Good table structure
            
            if 'edge_density' in region:
                edge_density = region['edge_density']
                if 0.02 <= edge_density <= 0.15:  # Good edge density for diagrams
                    score += 0.1
            
            # Window size bonus for sliding window detection
            if method == 'sliding_window':
                window_size = region.get('window_size', '')
                # Prefer certain window sizes that are good for anchor diagrams
                if any(size in window_size for size in ['200x300', '300x200', '250x400']):
                    score += 0.05
            
        except Exception:
            score = 0.1  # Default low score if calculation fails
        
        return min(1.0, max(0.0, score))  # Clamp to [0, 1]
    
    def _classify_visual_region(self, region_data: Dict, characteristic: str, debug: bool = False) -> Optional[Dict]:
        """Classify visual region for specific characteristic with enhanced construction-specific validation"""
        if characteristic not in self.training_data or not self.training_data[characteristic]:
            # If no training data, use content type heuristics
            return self._classify_by_content_type(region_data, characteristic, debug)
        
        # Calculate similarity to training examples
        similarities = []
        training_examples = self.training_data[characteristic]
        
        region_features = self._extract_visual_features(region_data['cv_image'])
        
        for example in training_examples:
            example_features = example.get('features', {})
            
            # Feature-based similarity
            feature_sim = self._calculate_feature_similarity(region_features, example_features)
            
            # Visual similarity (improved)
            visual_sim = self._calculate_visual_similarity(region_data['cv_image'], example['image'])
            
            # Combined similarity
            combined_sim = (feature_sim * 0.6) + (visual_sim * 0.4)
            similarities.append(combined_sim)
            
            if debug:
                print(f"        {example['filename']}: feature={feature_sim:.3f}, visual={visual_sim:.3f}, combined={combined_sim:.3f}")
        
        if not similarities:
            return None
        
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        # Enhanced classification logic with construction-specific validation
        base_threshold = 0.4  # Raised threshold to be more selective
        
        # Construction-specific content validation
        construction_validation = self._validate_construction_content(region_data, characteristic, debug)
        
        if not construction_validation['is_construction_related']:
            if debug:
                print(f"        Rejected: Not construction-related - {construction_validation['reason']}")
            return None
        
        # Boost confidence for certain content types
        confidence_boost = 1.0
        content_type = region_data.get('content_type', '')
        
        # Apply construction validation boost
        if construction_validation['confidence_boost'] > 0:
            confidence_boost += construction_validation['confidence_boost']
        
        # Tables and diagrams get preference for certain characteristics
        if content_type == 'table' and characteristic in ['design_pressure', 'impact_rating']:
            confidence_boost += 0.2
        elif content_type == 'diagram' and characteristic in ['anchors', 'glazing']:
            confidence_boost += 0.2
        
        # Apply detection method boost
        if region_data.get('confidence_boost', 0) > 0:
            confidence_boost += region_data['confidence_boost']
        
        final_similarity = min(1.0, max_similarity * confidence_boost)
        
        if debug:
            print(f"        Max similarity: {max_similarity:.3f}")
            print(f"        Construction validation: {construction_validation['is_construction_related']}")
            print(f"        Confidence boost: {confidence_boost:.2f}")
            print(f"        Final similarity: {final_similarity:.3f}")
            print(f"        Threshold: {base_threshold:.3f}")
        
        if final_similarity >= base_threshold:
            bbox = region_data['bbox']
            page_num = region_data['page']
            
            return {
                'type': characteristic,
                'content': f"{characteristic.replace('_', ' ').title()} - {content_type} from page {page_num}",
                'page': page_num,
                'confidence': final_similarity,
                'data_uri': self._pil_to_data_uri(region_data['image']),
                'extraction_id': f"{characteristic}_{page_num}_{int(time.time() * 1000) % 10000}",
                'bbox': bbox,
                'region_metadata': {
                    'detection_method': region_data.get('detection_method', 'unknown'),
                    'content_type': content_type,
                    'area': region_data.get('area', 0),
                    'visual_similarity_score': max_similarity,
                    'average_similarity': avg_similarity,
                    'confidence_boost_applied': confidence_boost,
                    'training_examples_used': len(similarities),
                    'construction_validation': construction_validation,
                    'similarity_method': 'enhanced_construction_validated_matching'
                }
            }
        
        return None
    
    def _validate_construction_content(self, region_data: Dict, characteristic: str, debug: bool = False) -> Dict:
        """Validate that content is construction-related and not logos/badges"""
        cv_region = region_data.get('cv_image')
        if cv_region is None:
            return {'is_construction_related': False, 'reason': 'No image data', 'confidence_boost': 0}
        
        try:
            gray = cv2.cvtColor(cv_region, cv2.COLOR_BGR2GRAY) if len(cv_region.shape) == 3 else cv_region
            height, width = gray.shape
            
            # Check for logo-like characteristics (things to avoid)
            logo_indicators = 0
            construction_indicators = 0
            
            # 1. Check for text density (logos often have large text)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Look for large connected components that might be text
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_text_areas = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (width * height) * 0.1:  # Large components (might be big text)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 5:  # Text-like aspect ratio
                        large_text_areas += 1
            
            if large_text_areas > 0:
                logo_indicators += 1
            
            # 2. Check for round/circular shapes (common in logos/badges)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min(width, height)//4,
                                     param1=50, param2=30, minRadius=min(width, height)//8, maxRadius=min(width, height)//3)
            
            if circles is not None and len(circles[0]) > 0:
                logo_indicators += 1
                if debug:
                    print(f"          Found {len(circles[0])} circular shapes (logo indicator)")
            
            # 3. Check for construction-related visual patterns
            edges = cv2.Canny(gray, 30, 100)
            
            # Look for line structures (common in technical drawings)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(width//4, 40), 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(height//4, 40)))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_pixels = np.sum(horizontal_lines > 0)
            v_line_pixels = np.sum(vertical_lines > 0)
            total_pixels = width * height
            
            line_density = (h_line_pixels + v_line_pixels) / total_pixels
            
            if line_density > 0.01:  # Has significant line structure
                construction_indicators += 1
                if debug:
                    print(f"          Line density: {line_density:.4f} (construction indicator)")
            
            # 4. Check for technical drawing characteristics
            # Look for dimension lines, arrows, or detailed drawings
            
            # Detect potential dimension lines (long horizontal/vertical lines)
            if h_line_pixels > 0:
                # Estimate average horizontal line length
                h_line_density = h_line_pixels / height if height > 0 else 0
                if h_line_density > width * 0.1:  # Long lines present
                    construction_indicators += 1
            
            if v_line_pixels > 0:
                # Estimate average vertical line length  
                v_line_density = v_line_pixels / width if width > 0 else 0
                if v_line_density > height * 0.1:  # Long lines present
                    construction_indicators += 1
            
            # 5. Size-based validation
            # Very small regions are more likely to be logos/badges
            area = width * height
            if area < 100000:  # Less than ~300x300 pixels
                logo_indicators += 1
                if debug:
                    print(f"          Small area: {area:,} pixels (logo indicator)")
            
            # 6. Aspect ratio check
            aspect_ratio = width / height if height > 0 else 0
            if 0.8 < aspect_ratio < 1.2:  # Square-ish (common for logos)
                logo_indicators += 0.5  # Partial indicator
            
            # Make decision
            is_construction_related = construction_indicators > logo_indicators
            
            # Calculate confidence boost
            confidence_boost = 0
            if is_construction_related:
                # Boost based on how strong the construction indicators are
                indicator_strength = construction_indicators - logo_indicators
                confidence_boost = min(0.3, indicator_strength * 0.1)  # Max 30% boost
            
            reason = f"Construction: {construction_indicators}, Logo: {logo_indicators}"
            
            if debug:
                print(f"          Construction validation: {is_construction_related}")
                print(f"          Reason: {reason}")
                print(f"          Confidence boost: {confidence_boost:.3f}")
            
            return {
                'is_construction_related': is_construction_related,
                'reason': reason,
                'confidence_boost': confidence_boost,
                'construction_indicators': construction_indicators,
                'logo_indicators': logo_indicators
            }
            
        except Exception as e:
            if debug:
                print(f"          Construction validation error: {e}")
            return {'is_construction_related': True, 'reason': f'Validation error: {e}', 'confidence_boost': 0}
    
    def _classify_by_content_type(self, region_data: Dict, characteristic: str, debug: bool = False) -> Optional[Dict]:
        """Fallback classification when no training data available"""
        content_type = region_data.get('content_type', '')
        
        # Heuristic matching based on content type and characteristic
        matches = False
        confidence = 0.4  # Base confidence for heuristic matching
        
        if characteristic == 'design_pressure':
            matches = content_type in ['table', 'visual_content']
            if content_type == 'table':
                confidence = 0.6
        elif characteristic == 'impact_rating':
            matches = content_type in ['table', 'visual_content']
            if content_type == 'table':
                confidence = 0.6
        elif characteristic in ['anchors', 'glazing']:
            matches = content_type in ['diagram', 'visual_content']
            if content_type == 'diagram':
                confidence = 0.5
        
        if matches and region_data.get('area', 0) > self.min_region_size * 1.5:
            bbox = region_data['bbox']
            page_num = region_data['page']
            
            if debug:
                print(f"        Heuristic match: {content_type} -> {characteristic} (confidence: {confidence:.3f})")
            
            return {
                'type': characteristic,
                'content': f"{characteristic.replace('_', ' ').title()} - {content_type} from page {page_num} (heuristic)",
                'page': page_num,
                'confidence': confidence,
                'data_uri': self._pil_to_data_uri(region_data['image']),
                'extraction_id': f"{characteristic}_{page_num}_{int(time.time() * 1000) % 10000}",
                'bbox': bbox,
                'region_metadata': {
                    'detection_method': region_data.get('detection_method', 'unknown'),
                    'content_type': content_type,
                    'area': region_data.get('area', 0),
                    'classification_method': 'heuristic_content_type',
                    'no_training_data': True
                }
            }
        
        return None
    
    def _extract_visual_features(self, cv_image) -> Dict:
        """Extract comprehensive visual features from image with detailed analysis"""
        if cv_image is None or cv_image.size == 0:
            return {}
        
        features = {}
        
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
            h, w = gray.shape
            
            # Basic features
            features['area'] = w * h
            features['aspect_ratio'] = w / h
            features['mean_intensity'] = np.mean(gray)
            features['intensity_std'] = np.std(gray)
            
            # Edge features (comprehensive)
            edges = cv2.Canny(gray, 30, 100)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Line features (detailed)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(w//4, 40), 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(h//4, 40)))
            
            horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            features['horizontal_lines'] = np.sum(horizontal > 0) / edges.size
            features['vertical_lines'] = np.sum(vertical > 0) / edges.size
            features['line_ratio'] = (features['horizontal_lines'] / features['vertical_lines']) if features['vertical_lines'] > 0 else 0
            
            # Content distribution
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            features['white_ratio'] = np.sum(binary > 127) / binary.size
            features['black_ratio'] = 1 - features['white_ratio']
            
        except Exception as e:
            # Return empty features if extraction fails
            features = {
                'area': w * h if 'w' in locals() and 'h' in locals() else 0,
                'aspect_ratio': w / h if 'w' in locals() and 'h' in locals() and h > 0 else 1,
                'feature_extraction_error': True
            }
        
        return features
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature sets"""
        if not features1 or not features2:
            return 0.0
        
        similarities = []
        
        # Compare numeric features
        numeric_features = ['edge_density', 'horizontal_lines', 'vertical_lines', 
                          'mean_intensity', 'intensity_std', 'white_ratio']
        
        for feature in numeric_features:
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                # Normalize differences to similarity
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(max(0.0, sim))
        
        # Compare aspect ratio
        if 'aspect_ratio' in features1 and 'aspect_ratio' in features2:
            ar1 = features1['aspect_ratio']
            ar2 = features2['aspect_ratio']
            ar_sim = 1.0 - min(1.0, abs(ar1 - ar2) / max(ar1, ar2))
            similarities.append(ar_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_visual_similarity(self, img1, img2) -> float:
        """Calculate comprehensive visual similarity using multiple methods"""
        try:
            # Resize to same size for comparison
            target_size = (256, 256)  # Larger size for better analysis
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)
            
            similarities = []
            
            # Method 1: HSV Histogram comparison
            hsv1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
            
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(max(0.0, hist_sim))
            
            # Method 2: Structural similarity using template matching
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            template_sim = max(0.0, min(1.0, max_val))
            similarities.append(template_sim)
            
            # Method 3: Edge similarity
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            edge_result = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
            _, edge_max, _, _ = cv2.minMaxLoc(edge_result)
            edge_sim = max(0.0, min(1.0, edge_max))
            similarities.append(edge_sim)
            
            # Weighted average of all similarities
            if similarities:
                weights = [0.4, 0.4, 0.2][:len(similarities)]
                weighted_sim = sum(sim * weight for sim, weight in zip(similarities, weights)) / sum(weights[:len(similarities)])
                return min(1.0, max(0.0, weighted_sim))
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _pil_to_data_uri(self, pil_image) -> str:
        """Convert PIL image to data URI"""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

# Utility functions for external use
def get_available_characteristics() -> List[str]:
    """Get available characteristics"""
    try:
        extractor = CharacteristicBasedExtractor()
        return extractor.get_available_characteristics()
    except Exception:
        return []

def get_characteristic_info(characteristic: str) -> Dict:
    """Get characteristic information"""
    try:
        extractor = CharacteristicBasedExtractor()
        return extractor.get_characteristic_info(characteristic)
    except Exception:
        return {}

def validate_labeled_data_structure() -> Dict:
    """Validate labeled data structure"""
    labeled_path = pathlib.Path("labeled_data")
    
    validation = {
        'labeled_data_exists': labeled_path.exists(),
        'matched_categories': [],
        'missing_categories': [],
        'recommendations': []
    }
    
    expected_categories = ['anchors', 'design_pressure', 'glazing', 'impact_rating']
    
    if not labeled_path.exists():
        validation['recommendations'].append("Run: python adaptive_agent.py --setup-labeled-data")
        validation['missing_categories'] = expected_categories
        return validation
    
    for category in expected_categories:
        category_path = labeled_path / category
        
        if category_path.exists():
            images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
            count = len(images)
            
            if count == 0:
                status = 'empty'
            elif count < 3:
                status = 'needs_more'
            else:
                status = 'good'
            
            validation['matched_categories'].append({
                'name': category,
                'count': count,
                'status': status
            })
            
            if status == 'empty':
                validation['recommendations'].append(f"Add training images to labeled_data/{category}/")
            elif status == 'needs_more':
                validation['recommendations'].append(f"Add more training images to labeled_data/{category}/ (need 3+)")
        else:
            validation['missing_categories'].append(category)
    
    if validation['missing_categories']:
        validation['recommendations'].append("Run: python adaptive_agent.py --setup-labeled-data")
    
    return validation
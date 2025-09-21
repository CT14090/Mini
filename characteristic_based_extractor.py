#!/usr/bin/env python3
# characteristic_based_extractor.py
"""
Characteristic-Based Content Extractor - CLEAN VERSION with Visual Similarity
Uses template matching instead of broken feature matching
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
    from PIL import Image
    CV_AVAILABLE = True
except ImportError:
    pass

class CharacteristicBasedExtractor:
    """Clean extractor with visual template matching"""
    
    def __init__(self):
        self.labeled_data_path = pathlib.Path("labeled_data")
        self.characteristics = self._load_characteristics()
        self.training_data = self._load_training_data()
        
        # Processing limits to prevent infinite loops
        self.max_regions_per_page = 5
        self.min_region_size = 10000
        self.processing_timeout_per_page = 15
        
        # For storing current region during classification
        self._current_cv_region = None
        
        print(f"✓ Extractor initialized with {len(self.characteristics)} characteristics")
    
    def _load_characteristics(self) -> Dict[str, Dict]:
        """Load available characteristics"""
        characteristics = {
            'anchors': {
                'name': 'Anchors',
                'description': 'Anchor details, attachment methods, and fastening systems',
                'keywords': ['anchor', 'fastener', 'attachment', 'bolt', 'screw'],
                'diagram_keywords': ['detail', 'section', 'assembly']
            },
            'design_pressure': {
                'name': 'Design Pressure', 
                'description': 'Design pressure ratings, wind load data, and structural calculations',
                'keywords': ['pressure', 'wind', 'load', 'psf', 'kpa'],
                'diagram_keywords': ['chart', 'table', 'rating', 'performance']
            },
            'glazing': {
                'name': 'Glazing',
                'description': 'Glass specifications, glazing details, and glazing systems',
                'keywords': ['glass', 'glazing', 'IG', 'IGU', 'pane'],
                'diagram_keywords': ['section', 'detail', 'specification']
            },
            'impact_rating': {
                'name': 'Impact Rating',
                'description': 'Impact resistance ratings, test results, and compliance data',
                'keywords': ['impact', 'missile', 'test', 'rating', 'zone'],
                'diagram_keywords': ['chart', 'table', 'certification', 'compliance']
            }
        }
        return characteristics
    
    def _load_training_data(self) -> Dict[str, List[Dict]]:
        """Load training data for visual comparison"""
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
            
            for img_file in image_files[:5]:  # Limit to 5 per category
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        training_data[category_name].append({
                            'image': img,
                            'path': str(img_file),
                            'filename': img_file.name
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
        """Extract content for specific characteristic"""
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
            
            # Extract regions
            regions = self._extract_regions_simple(cv_image, page_image, page_num, debug)
            
            if debug:
                print(f"    Found {len(regions)} potential regions")
            
            # Classify each region
            for i, region_data in enumerate(regions):
                # Check timeout
                if time.time() - start_time > self.processing_timeout_per_page:
                    if debug:
                        print(f"    Page timeout - processed {i}/{len(regions)} regions")
                    break
                
                classification = self._classify_region_for_characteristic(
                    region_data, characteristic, debug
                )
                
                if classification:
                    extracted_content.append(classification)
                    if debug:
                        print(f"      ✓ Classified: {classification['type']}")
                
                # Hard limit on extractions per page
                if len(extracted_content) >= self.max_regions_per_page:
                    if debug:
                        print(f"    Reached max extractions per page ({self.max_regions_per_page})")
                    break
            
            return extracted_content
            
        except Exception as e:
            if debug:
                print(f"    Error extracting content: {e}")
            return []
    
    def _extract_regions_simple(self, cv_image, pil_image: Image.Image, page_num: int, debug: bool = False) -> List[Dict]:
        """Extract regions using simple grid method"""
        regions = []
        height, width = cv_image.shape[:2]
        
        # Simple 3x3 grid
        grid_size = 3
        cell_w = width // grid_size
        cell_h = height // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * cell_w
                y = i * cell_h
                w = cell_w if j < grid_size - 1 else width - x
                h = cell_h if i < grid_size - 1 else height - y
                
                # Check minimum size
                if w * h >= self.min_region_size:
                    region_img = pil_image.crop((x, y, x + w, y + h))
                    
                    regions.append({
                        'image': region_img,
                        'cv_image': cv_image[y:y+h, x:x+w],
                        'bbox': (x, y, x + w, y + h),
                        'page': page_num,
                        'area': w * h,
                        'extraction_method': 'grid'
                    })
        
        return regions
    
    def _classify_region_for_characteristic(self, region_data: Dict, characteristic: str, debug: bool = False) -> Optional[Dict]:
        """Classify region using visual template matching with relative scoring"""
        region_image = region_data['image']
        cv_region = region_data.get('cv_image')
        
        # Store CV region for template matching
        self._current_cv_region = cv_region
        
        # Calculate visual similarity to ALL characteristics
        all_similarities = {}
        for char in self.characteristics.keys():
            if char in self.training_data and self.training_data[char]:
                similarity = self._calculate_similarity_to_characteristic({}, char)
                all_similarities[char] = similarity
        
        if not all_similarities:
            return None
        
        # Get similarity for target characteristic
        target_similarity = all_similarities.get(characteristic, 0.0)
        
        # Check if this characteristic is the BEST match among all categories
        best_char = max(all_similarities, key=all_similarities.get)
        best_similarity = all_similarities[best_char]
        
        if debug:
            print(f"        {characteristic}: visual_similarity={target_similarity:.3f}")
            print(f"        Best match: {best_char} ({best_similarity:.3f})")
            print(f"        All scores: {dict(sorted(all_similarities.items(), key=lambda x: x[1], reverse=True))}")
        
        # New criteria for acceptance:
        # 1. Must be the best match among all categories (or very close)
        # 2. Must have minimum absolute threshold (lowered significantly)
        # 3. Must look like technical content
        min_absolute_threshold = 0.05  # Much lower threshold
        relative_threshold = 0.8  # Must be 80% of best score
        
        is_best_match = (characteristic == best_char)
        is_close_to_best = (target_similarity >= best_similarity * relative_threshold)
        meets_minimum = (target_similarity >= min_absolute_threshold)
        is_technical = self._looks_like_technical_content(cv_region)
        
        if debug:
            print(f"        Is best match: {is_best_match}")
            print(f"        Close to best: {is_close_to_best}")
            print(f"        Meets minimum: {meets_minimum}")
            print(f"        Technical content: {is_technical}")
        
        if (is_best_match or is_close_to_best) and meets_minimum and is_technical:
            bbox = region_data['bbox']
            page_num = region_data['page']
            
            # Boost confidence if it's clearly the best match
            confidence_boost = 1.5 if is_best_match else 1.0
            final_confidence = min(1.0, target_similarity * confidence_boost)
            
            return {
                'type': characteristic,
                'content': f"{characteristic.replace('_', ' ').title()} from page {page_num}",
                'page': page_num,
                'confidence': final_confidence,
                'data_uri': self._pil_to_data_uri(region_image),
                'extraction_id': f"{characteristic}_{page_num}_{int(time.time() * 1000) % 10000}",
                'bbox': bbox,
                'region_metadata': {
                    'extraction_method': region_data.get('extraction_method', 'unknown'),
                    'area': region_data.get('area', 0),
                    'visual_similarity_score': target_similarity,
                    'best_match_score': best_similarity,
                    'all_similarities': all_similarities,
                    'is_technical_content': is_technical,
                    'characteristic': characteristic,
                    'similarity_method': 'visual_template_matching_relative'
                }
            }
        elif debug:
            if not meets_minimum:
                print(f"        Rejected: similarity {target_similarity:.3f} < {min_absolute_threshold}")
            if not (is_best_match or is_close_to_best):
                print(f"        Rejected: not best match for this region")
            if not is_technical:
                print(f"        Rejected: doesn't look like technical content")
        
        return None
    
    def _looks_like_technical_content(self, cv_region) -> bool:
        """Quick check if region looks like technical diagram vs text"""
        if cv_region is None:
            return False
        
        try:
            gray = cv2.cvtColor(cv_region, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Quick heuristics for technical content
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check for lines/shapes vs pure text
            white_ratio = np.sum(gray > 200) / (height * width)
            
            # Technical content typically has:
            # - Some edge density (but not too much - that's usually text)
            # - Not too much white space (pure text documents are mostly white)
            
            has_edges = edge_density > 0.02  # Some technical features
            not_too_much_white = white_ratio < 0.85  # Not a pure text document
            
            return has_edges and not_too_much_white
            
        except Exception:
            return True  # If we can't analyze, assume it might be technical
    
    def _calculate_similarity_to_characteristic(self, region_features: Dict, characteristic: str) -> float:
        """Calculate similarity using visual template matching"""
        if characteristic not in self.training_data:
            return 0.0
        
        training_examples = self.training_data[characteristic]
        if not training_examples:
            return 0.0
        
        # Get the current CV region for template matching
        cv_region = self._current_cv_region
        if cv_region is None:
            return 0.0
        
        similarities = []
        
        for training_example in training_examples:
            training_image = training_example.get('image')
            if training_image is None:
                continue
            
            # Use template matching
            template_sim = self._template_matching_similarity(cv_region, training_image)
            similarities.append(template_sim)
        
        return max(similarities) if similarities else 0.0
    
    def _template_matching_similarity(self, region_image, training_image) -> float:
        """Use OpenCV template matching for visual similarity"""
        try:
            # Resize both images to same size for comparison
            target_size = (200, 200)
            region_resized = cv2.resize(region_image, target_size)
            training_resized = cv2.resize(training_image, target_size)
            
            # Convert to grayscale
            region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
            training_gray = cv2.cvtColor(training_resized, cv2.COLOR_BGR2GRAY)
            
            # Template matching using normalized correlation
            result = cv2.matchTemplate(region_gray, training_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            # Ensure result is in [0, 1] range and add some noise tolerance
            similarity = max(0.0, min(1.0, max_val))
            
            # Boost similarity slightly if it's reasonable (helps with minor variations)
            if similarity > 0.1:
                similarity = min(1.0, similarity * 1.2)
            
            return similarity
            
        except Exception:
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
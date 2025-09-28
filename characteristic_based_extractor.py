#!/usr/bin/env python3
# characteristic_based_extractor.py
"""
Azure OpenAI-First Document Content Extractor - REVISED VERSION
Direct integration between Azure Vision and training data for accurate extraction
"""

import json
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple
import base64
import io

# Computer Vision imports (minimal usage)
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    CV_AVAILABLE = True
except ImportError:
    pass

# Azure OpenAI imports
AZURE_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import HumanMessage
    AZURE_AVAILABLE = True
except ImportError:
    print("Warning: Azure OpenAI not available - install with: pip install langchain-openai")

class AzureFirstExtractor:
    """Azure OpenAI-First extractor with direct training data integration"""
    
    def __init__(self):
        self.labeled_data_path = pathlib.Path("labeled_data")
        self.characteristics = self._load_characteristics()
        self.training_data = self._load_training_data()
        self.azure_client = None
        
        # Processing limits - optimized for Azure
        self.max_regions_per_page = 5  # Allow more findings
        self.min_region_size = 10000   # Smaller for details
        self.max_region_size = 3000000 # Larger for assemblies
        self.processing_timeout_per_page = 30
        
        # Initialize Azure OpenAI client
        if AZURE_AVAILABLE:
            self._init_azure_client()
        else:
            print("  Azure OpenAI not available - using fallback methods")
        
        print(f"Azure-First extractor initialized with {len(self.characteristics)} characteristics")
        if self.azure_client:
            print("  Azure OpenAI vision analysis: ENABLED")
        else:
            print("  Azure OpenAI vision analysis: DISABLED (using fallback)")
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client with testing"""
        try:
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
            
            if not all([endpoint, api_key, deployment]):
                print("  Azure OpenAI credentials incomplete - using fallback")
                return
            
            # Create the client
            self.azure_client = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                azure_deployment=deployment,
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=3000
            )
            
            print(f"  Azure OpenAI client initialized successfully")
                
        except Exception as e:
            print(f"  Azure OpenAI initialization failed: {e}")
            self.azure_client = None
    
    def _load_characteristics(self) -> Dict[str, Dict]:
        """Load characteristics with flexible descriptions for Azure"""
        characteristics = {
            'anchors': {
                'name': 'Anchors',
                'description': 'Anchor details, fastening systems, connection methods, bolt specifications',
                'flexible_terms': [
                    'anchor', 'fastener', 'bolt', 'screw', 'connection', 'attachment',
                    'mounting', 'fixing', 'hardware', 'connector', 'joint', 'coupling'
                ],
                'visual_patterns': [
                    'technical drawings of fasteners or connections',
                    'cross-section views showing attachment methods',
                    'detailed diagrams of bolts, screws, or anchors',
                    'assembly instructions with hardware',
                    'connection detail drawings',
                    'fastener specification tables or charts'
                ]
            },
            'design_pressure': {
                'name': 'Design Pressure',
                'description': 'Pressure ratings, wind loads, structural performance data',
                'flexible_terms': [
                    'pressure', 'wind', 'load', 'psf', 'kpa', 'rating', 'performance',
                    'structural', 'capacity', 'strength', 'resistance', 'force'
                ],
                'visual_patterns': [
                    'tables with pressure values or ratings',
                    'charts showing wind load data',
                    'performance matrices with numerical values',
                    'graphs with pressure or load information',
                    'rating tables with PSF, kPa, or similar units'
                ]
            },
            'glazing': {
                'name': 'Glazing',
                'description': 'Glass systems, glazing details, window specifications',
                'flexible_terms': [
                    'glass', 'glazing', 'window', 'pane', 'IG', 'IGU', 'insulated',
                    'double', 'triple', 'laminated', 'tempered', 'glazed'
                ],
                'visual_patterns': [
                    'cross-section drawings of glass assemblies',
                    'glazing system details',
                    'window construction diagrams',
                    'glass specification tables',
                    'IGU assembly drawings'
                ]
            },
            'impact_rating': {
                'name': 'Impact Rating',
                'description': 'Impact resistance, test results, compliance ratings',
                'flexible_terms': [
                    'impact', 'missile', 'test', 'rating', 'resistance', 'compliance',
                    'certification', 'zone', 'classification', 'standard', 'approval'
                ],
                'visual_patterns': [
                    'test result tables',
                    'impact rating charts',
                    'compliance matrices',
                    'certification tables',
                    'impact zone maps or classifications'
                ]
            }
        }
        return characteristics
    
    def _load_training_data(self) -> Dict[str, List[Dict]]:
        """Load training data with detailed analysis for Azure"""
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
            
            for img_file in image_files:
                try:
                    # Load PIL image
                    pil_img = Image.open(img_file)
                    
                    # Convert to data URI for Azure
                    data_uri = self._pil_to_data_uri(pil_img)
                    
                    # Basic properties
                    width, height = pil_img.size
                    
                    training_data[category_name].append({
                        'filename': img_file.name,
                        'path': str(img_file),
                        'data_uri': data_uri,
                        'dimensions': f"{width}x{height}",
                        'area': width * height
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
        """Extract content using Azure OpenAI Vision with enhanced prompting"""
        if characteristic not in self.characteristics:
            if debug:
                print(f"    Unknown characteristic: {characteristic}")
            return []
        
        # Primary method: Azure OpenAI Vision Analysis
        if self.azure_client and characteristic in self.training_data:
            extracted_content = self._extract_with_azure_vision_enhanced(
                page_image, characteristic, page_num, debug
            )
        else:
            # Fallback: Basic detection
            if debug:
                print(f"    Using fallback detection")
            extracted_content = self._extract_with_fallback(
                page_image, characteristic, page_num, debug
            )
        
        return extracted_content
    
    def _extract_with_azure_vision_enhanced(self, page_image: Image.Image, characteristic: str, 
                                           page_num: int, debug: bool = False) -> List[Dict]:
        """Enhanced Azure extraction with better prompting"""
        try:
            page_data_uri = self._pil_to_data_uri(page_image)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_extraction_prompt(characteristic, page_num)
            
            # Build message with training examples
            message_content = [{"type": "text", "text": prompt}]
            
            # Add the document page
            message_content.append({
                "type": "image_url", 
                "image_url": {"url": page_data_uri}
            })
            
            # Add training examples for reference
            training_examples = self.training_data.get(characteristic, [])
            if training_examples:
                message_content.append({
                    "type": "text", 
                    "text": f"\nTRAINING EXAMPLES ({len(training_examples)} examples of {characteristic}):"
                })
                
                for i, example in enumerate(training_examples[:3], 1):  # Show first 3
                    message_content.append({
                        "type": "text",
                        "text": f"\nExample {i} ({example['filename']}) - {example['dimensions']}:"
                    })
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": example['data_uri']}
                    })
            
            message = HumanMessage(content=message_content)
            
            if debug:
                print(f"    Sending enhanced prompt to Azure OpenAI...")
            
            # Get analysis
            response = self.azure_client.invoke([message])
            analysis_text = response.content
            
            if debug:
                print(f"    Azure response: {len(analysis_text)} chars")
                print(f"    Response preview: {analysis_text[:200]}...")
            
            # Parse response with better error handling
            extractions = self._parse_azure_response_enhanced(
                analysis_text, page_image, characteristic, page_num, debug
            )
            
            return extractions
            
        except Exception as e:
            if debug:
                print(f"    Azure extraction error: {e}")
            return self._extract_with_fallback(page_image, characteristic, page_num, debug)
    
    def _create_enhanced_extraction_prompt(self, characteristic: str, page_num: int) -> str:
        """Create flexible, permissive extraction prompt"""
        char_info = self.characteristics[characteristic]
        flexible_terms = char_info['flexible_terms']
        visual_patterns = char_info['visual_patterns']
        
        prompt = f"""
You are analyzing a construction document page for {characteristic.replace('_', ' ').upper()} content.

OBJECTIVE: Find ANY content related to {characteristic.replace('_', ' ')} - be INCLUSIVE and GENEROUS in your search.

WHAT TO LOOK FOR (be flexible with these terms):
{chr(10).join(f"• {term}" for term in flexible_terms)}

VISUAL CONTENT TO IDENTIFY:
{chr(10).join(f"• {pattern}" for pattern in visual_patterns)}

IMPORTANT INSTRUCTIONS:
1. Look at the TRAINING EXAMPLES I'll show you - find content that is similar or related
2. Be GENEROUS in your interpretation - if something might be related to {characteristic.replace('_', ' ')}, include it
3. Don't be overly restrictive - I'd rather have false positives than miss relevant content
4. Look for diagrams, tables, drawings, specifications, or any visual content related to the topic
5. Even if you're not 100% certain, if it seems related, include it

RESPONSE FORMAT:
For each relevant region you find, respond with:

REGION_[NUMBER]:
DESCRIPTION: What you see in this region
COORDINATES: [x, y, width, height] in pixels (estimate the bounding box)
CONFIDENCE: Your confidence from 0.1 to 1.0 (be generous - use 0.5+ if it might be relevant)
REASONING: Why you think this relates to {characteristic.replace('_', ' ')}

If you find NO relevant content after careful examination, respond with: "NO_RELEVANT_CONTENT"

Remember: Be INCLUSIVE rather than exclusive. I want to capture all potentially relevant content.

ANALYZE THE DOCUMENT PAGE NOW:
"""
        return prompt
    
    def _parse_azure_response_enhanced(self, response_text: str, page_image: Image.Image, 
                                     characteristic: str, page_num: int, debug: bool = False) -> List[Dict]:
        """Enhanced response parsing with better error handling"""
        extractions = []
        
        if "NO_RELEVANT_CONTENT" in response_text.upper():
            if debug:
                print(f"    Azure explicitly found no relevant content")
            return extractions
        
        # Parse regions more flexibly
        lines = response_text.split('\n')
        current_region = {}
        
        for line in lines:
            line = line.strip()
            
            if 'REGION_' in line and ':' in line:
                # Save previous region
                if current_region:
                    extraction = self._create_extraction_from_region_enhanced(
                        current_region, page_image, characteristic, page_num, debug
                    )
                    if extraction:
                        extractions.append(extraction)
                
                current_region = {'region_line': line}
            
            elif current_region and ':' in line:
                key_value = line.split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip().lower()
                    value = key_value[1].strip()
                    
                    if 'description' in key:
                        current_region['description'] = value
                    elif 'coordinates' in key or 'coordinate' in key:
                        current_region['coordinates_str'] = value
                    elif 'confidence' in key:
                        try:
                            current_region['confidence'] = float(value)
                        except:
                            current_region['confidence'] = 0.6  # Default generous confidence
                    elif 'reasoning' in key:
                        current_region['reasoning'] = value
        
        # Don't forget the last region
        if current_region:
            extraction = self._create_extraction_from_region_enhanced(
                current_region, page_image, characteristic, page_num, debug
            )
            if extraction:
                extractions.append(extraction)
        
        if debug:
            print(f"    Parsed {len(extractions)} regions from Azure response")
        
        return extractions
    
    def _create_extraction_from_region_enhanced(self, region_data: Dict, page_image: Image.Image, 
                                              characteristic: str, page_num: int, debug: bool = False) -> Optional[Dict]:
        """Create extraction with flexible coordinate parsing"""
        try:
            page_width, page_height = page_image.size
            
            # Parse coordinates flexibly
            coords_str = region_data.get('coordinates_str', '')
            coords = self._parse_coordinates_flexible(coords_str, page_width, page_height)
            
            if not coords:
                # If no coordinates, create a default region (center portion of page)
                x = page_width // 4
                y = page_height // 4
                w = page_width // 2
                h = page_height // 2
                coords = [x, y, w, h]
                if debug:
                    print(f"    Using default coordinates: {coords}")
            
            x, y, w, h = coords
            
            # Validate and adjust coordinates
            x = max(0, min(x, page_width - 1))
            y = max(0, min(y, page_height - 1))
            w = max(50, min(w, page_width - x))  # Minimum 50px width
            h = max(50, min(h, page_height - y))  # Minimum 50px height
            
            area = w * h
            
            # More lenient size checking
            if area < 1000:  # Too small
                return None
            
            # Extract region
            try:
                region_img = page_image.crop((x, y, x + w, y + h))
            except:
                # If crop fails, use a safe region
                region_img = page_image.crop((0, 0, min(200, page_width), min(200, page_height)))
            
            # Create extraction
            extraction = {
                'type': characteristic,
                'content': region_data.get('description', f"{characteristic.replace('_', ' ').title()} content from page {page_num}"),
                'page': page_num,
                'confidence': region_data.get('confidence', 0.6),
                'data_uri': self._pil_to_data_uri(region_img),
                'extraction_id': f"{characteristic}_{page_num}_{int(time.time() * 1000) % 10000}",
                'bbox': (x, y, x + w, y + h),
                'region_metadata': {
                    'detection_method': 'azure_openai_vision_enhanced',
                    'content_type': characteristic,
                    'area': area,
                    'azure_description': region_data.get('description', ''),
                    'azure_reasoning': region_data.get('reasoning', ''),
                    'coordinates_source': 'azure_vision_analysis',
                    'extraction_method': 'azure_enhanced_prompting'
                }
            }
            
            return extraction
            
        except Exception as e:
            if debug:
                print(f"    Error creating extraction: {e}")
            return None
    
    def _parse_coordinates_flexible(self, coords_str: str, page_width: int, page_height: int) -> Optional[List[int]]:
        """Flexible coordinate parsing"""
        if not coords_str:
            return None
        
        try:
            # Remove brackets and extra characters
            coords_str = coords_str.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            
            # Try to parse numbers
            numbers = []
            for part in coords_str.split(','):
                part = part.strip()
                # Extract numbers from the part
                import re
                number_match = re.search(r'\d+', part)
                if number_match:
                    numbers.append(int(number_match.group()))
            
            if len(numbers) >= 4:
                x, y, w, h = numbers[:4]
                
                # Validate coordinates make sense
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    if x < page_width and y < page_height:
                        return [x, y, min(w, page_width - x), min(h, page_height - y)]
            
        except:
            pass
        
        return None
    
    def _extract_with_fallback(self, page_image: Image.Image, characteristic: str, 
                              page_num: int, debug: bool = False) -> List[Dict]:
        """Simple fallback when Azure unavailable"""
        if not CV_AVAILABLE:
            return []
        
        if debug:
            print(f"    Running fallback detection")
        
        # Simple region detection
        cv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        height, width = cv_image.shape[:2]
        
        # Create a few sample regions
        regions = []
        
        # Center region
        center_x = width // 4
        center_y = height // 4
        center_w = width // 2
        center_h = height // 2
        
        region_img = page_image.crop((center_x, center_y, center_x + center_w, center_y + center_h))
        
        regions.append({
            'type': characteristic,
            'content': f"Fallback {characteristic.replace('_', ' ')} detection from page {page_num}",
            'page': page_num,
            'confidence': 0.3,  # Lower confidence for fallback
            'data_uri': self._pil_to_data_uri(region_img),
            'extraction_id': f"{characteristic}_{page_num}_fallback",
            'bbox': (center_x, center_y, center_x + center_w, center_y + center_h),
            'region_metadata': {
                'detection_method': 'fallback_detection',
                'content_type': 'potential_content',
                'area': center_w * center_h,
                'extraction_method': 'simple_fallback'
            }
        })
        
        return regions
    
    def _pil_to_data_uri(self, pil_image) -> str:
        """Convert PIL image to data URI"""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

# Utility functions and compatibility
def get_available_characteristics() -> List[str]:
    try:
        extractor = AzureFirstExtractor()
        return extractor.get_available_characteristics()
    except Exception:
        return []

def get_characteristic_info(characteristic: str) -> Dict:
    try:
        extractor = AzureFirstExtractor()
        return extractor.get_characteristic_info(characteristic)
    except Exception:
        return {}

def validate_labeled_data_structure() -> Dict:
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
            
            if count >= 3:
                status = 'excellent'
            elif count >= 1:
                status = 'good'
            else:
                status = 'empty'
            
            validation['matched_categories'].append({
                'name': category,
                'count': count,
                'status': status
            })
            
            if status == 'empty':
                validation['recommendations'].append(f"Add training images to labeled_data/{category}/")
        else:
            validation['missing_categories'].append(category)
    
    return validation

# Compatibility alias for backward compatibility
CharacteristicBasedExtractor = AzureFirstExtractor
#!/usr/bin/env python3
"""
Specialized Window Characteristics Extraction Agent
Focuses on specific window characteristics: anchors, glazing, impact ratings, design pressure
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

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
    from PIL import Image
    CV_AVAILABLE = True
    print("✓ Computer Vision available for characteristic detection")
except ImportError:
    print("⚠ OpenCV/PIL not available - using standard extraction")

# Docling imports
DOCLING_OK = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
    DOCLING_OK = True
    print("✓ Docling document processing available")
except Exception as e:
    print(f"✗ Docling import failed: {e}")

class WindowCharacteristic(Enum):
    """Window characteristics to extract"""
    ANCHORS = "anchors"
    GLAZING = "glazing"
    IMPACT_RATING = "impact_rating"
    DESIGN_PRESSURE = "design_pressure"

@dataclass
class CharacteristicParameters:
    """Parameters specific to each characteristic"""
    confidence_threshold: float = 0.4
    min_section_length: int = 150
    max_extractions: int = 20
    image_size_min: int = 150
    table_relevance_threshold: int = 2
    cv_detection_threshold: int = 40
    content_classification_threshold: float = 0.3
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class WindowCharacteristicAgent:
    """Base agent for extracting window characteristics"""
    
    def __init__(self, characteristic: WindowCharacteristic):
        self.characteristic = characteristic
        print(f"🔧 Initializing {characteristic.value.title()} Extraction Agent...")
        
        self.params = self._load_characteristic_parameters()
        print(f"✓ Loaded {characteristic.value} parameters")
        
        # Initialize document converter
        if DOCLING_OK:
            pp = PdfPipelineOptions()
            pp.images_scale = 2.8
            pp.generate_page_images = True
            pp.generate_picture_images = True
            pp.generate_table_images = True
            
            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pp)}
            )
        else:
            raise RuntimeError("Docling required but not available")
        
        # Define characteristic-specific keywords and patterns
        self.characteristic_config = self._get_characteristic_config()
        self._check_azure_config()
    
    def _get_characteristic_config(self) -> Dict:
        """Get configuration for specific characteristic"""
        configs = {
            WindowCharacteristic.ANCHORS: {
                'keywords': [
                    'anchor', 'anchors', 'anchoring', 'fastener', 'fasteners',
                    'concrete screw', 'wood screw', 'buck', '1by buck', '2by buck',
                    'self drilling', 'self-drilling', 'metal structure', 'structural screw',
                    'concrete anchor', 'wood anchor', 'masonry anchor'
                ],
                'anchor_types': [
                    'directly into concrete', 'directly into wood', 'into wood via 1by buck',
                    'into concrete via 1by buck', 'into concrete via 2by buck',
                    'self drilling screws into metal', 'masonry anchor', 'structural anchor'
                ],
                'table_indicators': ['anchor', 'fastener', 'screw', 'attachment'],
                'image_patterns': ['anchor', 'screw', 'fastener', 'attachment detail'],
                'context_terms': ['installation', 'attachment', 'mounting', 'fastening']
            },
            
            WindowCharacteristic.GLAZING: {
                'keywords': [
                    'glazing', 'glass', 'igp', 'igu', 'insulated glass',
                    'low-e', 'low e', 'laminated', 'tempered', 'annealed',
                    'thickness', 'mm', 'inch', 'double glazed', 'triple glazed'
                ],
                'glass_types': [
                    'tempered', 'laminated', 'annealed', 'low-e', 'insulated glass',
                    'double glazed', 'triple glazed', 'safety glass', 'impact glass'
                ],
                'table_indicators': ['glass', 'glazing', 'thickness', 'type', 'specification'],
                'image_patterns': ['glass detail', 'glazing section', 'glass assembly'],
                'context_terms': ['optical', 'thermal', 'acoustic', 'performance']
            },
            
            WindowCharacteristic.IMPACT_RATING: {
                'keywords': [
                    'impact', 'missile', 'impact rating', 'impact resistance',
                    'small missile', 'large missile', 'both missile',
                    'astm', 'aama', 'hurricane', 'storm', 'debris'
                ],
                'impact_types': [
                    'small missile impact', 'large missile impact', 'both missile impact',
                    'impact resistant', 'hurricane rated', 'storm rated'
                ],
                'table_indicators': ['impact', 'missile', 'rating', 'resistance', 'test'],
                'image_patterns': ['impact test', 'missile impact', 'test result'],
                'context_terms': ['test', 'certification', 'compliance', 'rating']
            },
            
            WindowCharacteristic.DESIGN_PRESSURE: {
                'keywords': [
                    'design pressure', 'dp', 'pressure rating', 'wind load',
                    'psf', 'pa', 'kpa', 'positive', 'negative', 'structural',
                    'load rating', 'wind rating', 'pressure test'
                ],
                'pressure_types': [
                    'positive pressure', 'negative pressure', 'design pressure',
                    'wind load rating', 'structural load', 'pressure rating'
                ],
                'table_indicators': ['pressure', 'dp', 'load', 'rating', 'psf', 'pa'],
                'image_patterns': ['pressure table', 'load table', 'rating table'],
                'context_terms': ['structural', 'wind', 'load', 'performance']
            }
        }
        
        return configs.get(self.characteristic, configs[WindowCharacteristic.ANCHORS])
    
    def _load_characteristic_parameters(self) -> CharacteristicParameters:
        """Load parameters specific to characteristic"""
        param_file = pathlib.Path(f"parameters_{self.characteristic.value}.json")
        if param_file.exists():
            try:
                with open(param_file) as f:
                    data = json.load(f)
                return CharacteristicParameters.from_dict(data)
            except Exception as e:
                print(f"⚠ Error loading {self.characteristic.value} parameters: {e}")
        
        # Create default parameters
        default_params = CharacteristicParameters()
        self._save_parameters(default_params)
        return default_params
    
    def _save_parameters(self, params: CharacteristicParameters = None):
        """Save characteristic-specific parameters"""
        if params is None:
            params = self.params
        param_file = f"parameters_{self.characteristic.value}.json"
        with open(param_file, 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def _check_azure_config(self):
        """Check Azure OpenAI configuration"""
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠ Azure OpenAI missing: {missing_vars}")
            print("  Will use fallback analysis")
        else:
            print("✓ Azure OpenAI configured")
    
    def process_document(self, source: str, debug: bool = False) -> str:
        """Process document focusing on specific characteristic"""
        start_time = time.time()
        doc_id = hashlib.md5(f"{source}_{self.characteristic.value}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        print(f"\n{'='*70}")
        print(f"🎯 {self.characteristic.value.upper()} EXTRACTION AGENT")
        print(f"{'='*70}")
        print(f"📄 Source: {os.path.basename(source)}")
        print(f"🎯 Focus: {self.characteristic.value.replace('_', ' ').title()}")
        print(f"🆔 Document ID: {doc_id}")
        print(f"⚙️  Parameters:")
        print(f"   • Confidence: {self.params.confidence_threshold}")
        print(f"   • Min Size: {self.params.image_size_min}px")
        print(f"   • Max Items: {self.params.max_extractions}")
        print(f"{'='*70}")
        
        # Download if URL
        if source.startswith(("http://", "https://")):
            print("🌐 Downloading document...")
            source = self._download(source)
        
        # Convert document
        print("📖 Converting document with Docling...")
        conv_res = self.converter.convert(source)
        doc = conv_res.document
        
        # Extract all elements
        all_elements = list(doc.iterate_items())
        print(f"✓ Found {len(all_elements)} document elements")
        
        # Extract characteristic-specific content
        extracted_content = []
        
        print(f"\n🔍 EXTRACTING {self.characteristic.value.upper()} CONTENT")
        print(f"{'─'*50}")
        
        # Extract images related to characteristic
        characteristic_images = self._extract_characteristic_images(doc, all_elements, debug)
        print(f"✓ Found {len(characteristic_images)} relevant images")
        
        # Extract tables related to characteristic
        characteristic_tables = self._extract_characteristic_tables(all_elements, debug)
        print(f"✓ Found {len(characteristic_tables)} relevant tables")
        
        # Extract text sections related to characteristic
        characteristic_text = self._extract_characteristic_text(all_elements, debug)
        print(f"✓ Found {len(characteristic_text)} relevant text sections")
        
        # Combine all extractions
        extracted_content.extend(characteristic_images)
        extracted_content.extend(characteristic_tables)
        extracted_content.extend(characteristic_text)
        
        # Sort by confidence and limit
        extracted_content.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        if len(extracted_content) > self.params.max_extractions:
            extracted_content = extracted_content[:self.params.max_extractions]
        
        # Generate detailed analysis
        analysis = self._analyze_extractions(extracted_content)
        
        total_items = len(extracted_content)
        processing_time = time.time() - start_time
        
        print(f"\n📊 {self.characteristic.value.upper()} EXTRACTION SUMMARY")
        print(f"{'─'*50}")
        print(f"Total items extracted: {total_items}")
        print(f"Images: {len(characteristic_images)}")
        print(f"Tables: {len(characteristic_tables)}")
        print(f"Text sections: {len(characteristic_text)}")
        print(f"Processing time: {processing_time:.1f}s")
        
        if analysis:
            print(f"\n🎯 {self.characteristic.value.upper()} ANALYSIS:")
            for key, value in analysis.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Save extraction data
        extraction_data = self._create_extraction_data(
            doc_id, source, extracted_content, processing_time, analysis
        )
        
        pathlib.Path("feedback_data").mkdir(exist_ok=True)
        extraction_file = f"feedback_data/{self.characteristic.value}_extraction_{doc_id}.json"
        with open(extraction_file, 'w') as f:
            json.dump(extraction_data, f, indent=2)
        
        print(f"💾 Extraction saved: {extraction_file}")
        
        # Run LLM feedback
        print(f"\n🤖 RUNNING {self.characteristic.value.upper()} LLM FEEDBACK")
        print(f"{'─'*50}")
        success = self._run_llm_feedback(doc_id, debug)
        if success:
            print("✅ LLM feedback completed")
        else:
            print("⚠️  Used fallback analysis")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"🎯 {self.characteristic.value.upper()} EXTRACTION COMPLETED")
        print(f"{'='*70}")
        print(f"Document ID: {doc_id}")
        print(f"Items extracted: {total_items}")
        print(f"Focus: {self.characteristic.value.replace('_', ' ').title()}")
        print(f"View results: streamlit run feedback_interface.py")
        print(f"{'='*70}")
        
        return doc_id
    
    def _extract_characteristic_images(self, doc, all_elements, debug: bool = False) -> List[Dict]:
        """Extract images related to specific characteristic"""
        characteristic_images = []
        picture_items = [element for element, _ in all_elements if isinstance(element, PictureItem)]
        config = self.characteristic_config
        
        for i, element in enumerate(picture_items):
            try:
                pil_image = element.get_image(doc)
                if not pil_image:
                    continue
                
                width, height = pil_image.size
                page_no = self._get_page_number(element)
                area = width * height
                
                # Size filtering
                if area < (self.params.image_size_min ** 2):
                    continue
                
                # Get surrounding context
                context = self._extract_page_context(all_elements, page_no)
                
                # Check relevance to characteristic
                relevance_score = self._calculate_image_relevance(context, config, debug)
                
                if debug:
                    print(f"    Image {i+1}: {width}x{height}, page {page_no}, relevance: {relevance_score:.3f}")
                
                if relevance_score >= self.params.content_classification_threshold:
                    # Further analysis if CV available
                    cv_analysis = {}
                    if CV_AVAILABLE:
                        cv_analysis = self._analyze_image_for_characteristic(pil_image, config, debug)
                        relevance_score += cv_analysis.get('cv_boost', 0)
                    
                    characteristic_images.append({
                        'type': f'{self.characteristic.value}_image',
                        'content': f"{self.characteristic.value.replace('_', ' ').title()} image from page {page_no}",
                        'page': page_no,
                        'confidence': min(relevance_score, 1.0),
                        'data_uri': self._pil_to_data_uri(pil_image),
                        'extraction_id': f"{self.characteristic.value}_img_{page_no}_{i}",
                        'context': context,
                        'cv_analysis': cv_analysis,
                        'metadata': {
                            'width': width,
                            'height': height,
                            'area': area,
                            'characteristic': self.characteristic.value
                        }
                    })
                    
                    if debug:
                        print(f"        ✓ Extracted as {self.characteristic.value} image")
            
            except Exception as e:
                if debug:
                    print(f"    Error processing image {i+1}: {e}")
                continue
        
        return characteristic_images
    
    def _extract_characteristic_tables(self, all_elements, debug: bool = False) -> List[Dict]:
        """Extract tables related to specific characteristic"""
        characteristic_tables = []
        table_items = [element for element, _ in all_elements if isinstance(element, TableItem)]
        config = self.characteristic_config
        
        for i, element in enumerate(table_items):
            try:
                content = element.export_to_markdown()
                page_num = self._get_page_number(element)
                
                if len(content.strip()) < self.params.min_section_length:
                    continue
                
                # Check table relevance to characteristic
                relevance_score = self._calculate_table_relevance(content, config, debug)
                
                if debug:
                    print(f"    Table {i+1} (page {page_num}): {len(content)} chars, relevance: {relevance_score:.3f}")
                
                if relevance_score >= self.params.content_classification_threshold:
                    # Analyze table content for specific characteristic data
                    table_analysis = self._analyze_table_for_characteristic(content, config)
                    
                    characteristic_tables.append({
                        'type': f'{self.characteristic.value}_table',
                        'content': content,
                        'page': page_num,
                        'confidence': min(relevance_score, 1.0),
                        'extraction_id': f"{self.characteristic.value}_table_{page_num}_{i}",
                        'table_analysis': table_analysis,
                        'metadata': {
                            'row_count': content.count('\n'),
                            'characteristic': self.characteristic.value,
                            'content_length': len(content)
                        }
                    })
                    
                    if debug:
                        print(f"        ✓ Extracted as {self.characteristic.value} table")
            
            except Exception as e:
                if debug:
                    print(f"    Error processing table {i+1}: {e}")
                continue
        
        return characteristic_tables
    
    def _extract_characteristic_text(self, all_elements, debug: bool = False) -> List[Dict]:
        """Extract text sections related to specific characteristic"""
        characteristic_text = []
        config = self.characteristic_config
        
        # Group text by page
        page_texts = {}
        for element, _ in all_elements:
            # Handle both Docling TextItem and fallback elements
            is_text_element = (isinstance(element, TextItem) if not self.use_fallback 
                             else hasattr(element, 'text'))
            
            if is_text_element:
                page_num = self._get_page_number(element)
                text_content = element.text if hasattr(element, 'text') else str(element)
                
                if page_num not in page_texts:
                    page_texts[page_num] = []
                page_texts[page_num].append(text_content)
        
        # Process each page's text
        for page_num, texts in page_texts.items():
            combined_text = ' '.join(texts)
            
            if len(combined_text.strip()) < self.params.min_section_length:
                continue
            
            # Check text relevance to characteristic
            relevance_score = self._calculate_text_relevance(combined_text, config, debug)
            
            if debug and relevance_score > 0:
                print(f"    Page {page_num} text: {len(combined_text)} chars, relevance: {relevance_score:.3f}")
            
            if relevance_score >= self.params.content_classification_threshold:
                # Extract specific characteristic mentions
                characteristic_mentions = self._extract_characteristic_mentions(combined_text, config)
                
                characteristic_text.append({
                    'type': f'{self.characteristic.value}_text',
                    'content': combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text,
                    'page': page_num,
                    'confidence': min(relevance_score, 1.0),
                    'extraction_id': f"{self.characteristic.value}_text_{page_num}",
                    'characteristic_mentions': characteristic_mentions,
                    'metadata': {
                        'full_length': len(combined_text),
                        'characteristic': self.characteristic.value,
                        'mentions_count': len(characteristic_mentions)
                    }
                })
                
                if debug:
                    print(f"        ✓ Extracted as {self.characteristic.value} text")
        
        return characteristic_text
    
    def _calculate_image_relevance(self, context: Dict, config: Dict, debug: bool = False) -> float:
        """Calculate image relevance to characteristic"""
        page_text = context.get('page_text', '').lower()
        score = 0.0
        
        # Check for characteristic keywords in context
        keyword_matches = 0
        for keyword in config['keywords']:
            if keyword.lower() in page_text:
                keyword_matches += 1
                score += 0.1
        
        # Check for image pattern indicators
        for pattern in config['image_patterns']:
            if pattern.lower() in page_text:
                score += 0.2
        
        # Check for context terms
        for term in config['context_terms']:
            if term.lower() in page_text:
                score += 0.05
        
        if debug and score > 0:
            print(f"          Image relevance: {keyword_matches} keywords, score: {score:.3f}")
        
        return min(score, 1.0)
    
    def _calculate_table_relevance(self, content: str, config: Dict, debug: bool = False) -> float:
        """Calculate table relevance to characteristic"""
        content_lower = content.lower()
        score = 0.0
        
        # Check for table indicators
        indicator_matches = 0
        for indicator in config['table_indicators']:
            if indicator.lower() in content_lower:
                indicator_matches += 1
                score += 0.15
        
        # Check for characteristic keywords
        keyword_matches = 0
        for keyword in config['keywords']:
            if keyword.lower() in content_lower:
                keyword_matches += 1
                score += 0.1
        
        # Check for specific characteristic types
        for char_type in config.get(f"{self.characteristic.value.split('_')[0]}_types", []):
            if char_type.lower() in content_lower:
                score += 0.25
        
        if debug and score > 0:
            print(f"          Table relevance: {indicator_matches} indicators, {keyword_matches} keywords, score: {score:.3f}")
        
        return min(score, 1.0)
    
    def _calculate_text_relevance(self, text: str, config: Dict, debug: bool = False) -> float:
        """Calculate text relevance to characteristic"""
        text_lower = text.lower()
        score = 0.0
        
        # Check for characteristic keywords (weighted by frequency)
        keyword_matches = 0
        for keyword in config['keywords']:
            count = text_lower.count(keyword.lower())
            if count > 0:
                keyword_matches += count
                score += min(count * 0.1, 0.3)  # Cap per keyword
        
        # Check for specific characteristic types
        for char_type in config.get(f"{self.characteristic.value.split('_')[0]}_types", []):
            if char_type.lower() in text_lower:
                score += 0.3
        
        # Bonus for concentrated mentions
        if keyword_matches >= 3:
            score += 0.2
        
        if debug and score > 0:
            print(f"          Text relevance: {keyword_matches} keyword matches, score: {score:.3f}")
        
        return min(score, 1.0)
    
    def _analyze_image_for_characteristic(self, pil_image, config: Dict, debug: bool = False) -> Dict:
        """Analyze image using computer vision for characteristic-specific features"""
        try:
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            analysis = {
                'cv_boost': 0.0,
                'features_detected': []
            }
            
            # Characteristic-specific CV analysis
            if self.characteristic == WindowCharacteristic.ANCHORS:
                # Look for circular/hex shapes (screws/bolts)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=5, maxRadius=50)
                if circles is not None and len(circles[0]) > 2:
                    analysis['cv_boost'] += 0.2
                    analysis['features_detected'].append(f"anchor_points: {len(circles[0])}")
            
            elif self.characteristic == WindowCharacteristic.GLAZING:
                # Look for parallel lines (glass layers)
                lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=30,
                                      minLineLength=50, maxLineGap=10)
                if lines is not None and len(lines) > 4:
                    analysis['cv_boost'] += 0.15
                    analysis['features_detected'].append(f"glass_lines: {len(lines)}")
            
            elif self.characteristic == WindowCharacteristic.DESIGN_PRESSURE:
                # Look for tabular structure
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
                if cv2.countNonZero(horizontal_lines) > 100:
                    analysis['cv_boost'] += 0.1
                    analysis['features_detected'].append("tabular_structure")
            
            return analysis
            
        except Exception as e:
            if debug:
                print(f"          CV analysis error: {e}")
            return {'cv_boost': 0.0, 'features_detected': []}
    
    def _analyze_table_for_characteristic(self, content: str, config: Dict) -> Dict:
        """Analyze table content for characteristic-specific data"""
        analysis = {
            'data_points_found': [],
            'characteristic_specific_data': {}
        }
        
        content_lower = content.lower()
        
        if self.characteristic == WindowCharacteristic.ANCHORS:
            # Look for anchor specifications
            anchor_types = config.get('anchor_types', [])
            for anchor_type in anchor_types:
                if anchor_type.lower() in content_lower:
                    analysis['data_points_found'].append(anchor_type)
            
            # Look for screw sizes, materials
            import re
            screw_sizes = re.findall(r'#\d+|M\d+|\d+mm|\d+inch', content)
            if screw_sizes:
                analysis['characteristic_specific_data']['screw_sizes'] = screw_sizes
        
        elif self.characteristic == WindowCharacteristic.GLAZING:
            # Look for glass specifications
            import re
            thicknesses = re.findall(r'\d+(?:\.\d+)?\s*mm|\d+/\d+\s*inch', content)
            if thicknesses:
                analysis['characteristic_specific_data']['glass_thicknesses'] = thicknesses
            
            glass_types = config.get('glass_types', [])
            for glass_type in glass_types:
                if glass_type.lower() in content_lower:
                    analysis['data_points_found'].append(glass_type)
        
        elif self.characteristic == WindowCharacteristic.IMPACT_RATING:
            # Look for impact ratings
            impact_types = config.get('impact_types', [])
            for impact_type in impact_types:
                if impact_type.lower() in content_lower:
                    analysis['data_points_found'].append(impact_type)
        
        elif self.characteristic == WindowCharacteristic.DESIGN_PRESSURE:
            # Look for pressure values
            import re
            pressure_values = re.findall(r'\d+(?:\.\d+)?\s*(?:psf|pa|kpa)', content_lower)
            if pressure_values:
                analysis['characteristic_specific_data']['pressure_values'] = pressure_values
        
        return analysis
    
    def _extract_characteristic_mentions(self, text: str, config: Dict) -> List[str]:
        """Extract specific mentions of characteristic data"""
        mentions = []
        text_lower = text.lower()
        
        # Extract sentences containing characteristic keywords
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for keyword in config['keywords'][:10]:  # Top keywords only
                if keyword.lower() in sentence_lower and len(sentence_lower) > 20:
                    mentions.append(sentence.strip())
                    break
        
        return mentions[:5]  # Limit to 5 most relevant mentions
    
    def _analyze_extractions(self, extracted_content: List[Dict]) -> Dict:
        """Analyze the extractions for characteristic-specific insights"""
        analysis = {
            'total_items': len(extracted_content),
            'avg_confidence': 0.0,
            'characteristic_coverage': 'unknown'
        }
        
        if not extracted_content:
            return analysis
        
        # Calculate average confidence
        confidences = [item.get('confidence', 0) for item in extracted_content]
        analysis['avg_confidence'] = sum(confidences) / len(confidences)
        
        # Count by type
        type_counts = {}
        for item in extracted_content:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        analysis['content_types'] = type_counts
        
        # Characteristic-specific analysis
        if self.characteristic == WindowCharacteristic.ANCHORS:
            # Count different anchor types found
            anchor_types_found = set()
            for item in extracted_content:
                data_points = item.get('table_analysis', {}).get('data_points_found', [])
                anchor_types_found.update(data_points)
            analysis['anchor_types_identified'] = len(anchor_types_found)
            analysis['coverage'] = 'comprehensive' if len(anchor_types_found) >= 3 else 'partial'
        
        elif self.characteristic == WindowCharacteristic.GLAZING:
            # Analyze glazing information completeness
            glazing_data = []
            for item in extracted_content:
                char_data = item.get('table_analysis', {}).get('characteristic_specific_data', {})
                if 'glass_thicknesses' in char_data:
                    glazing_data.extend(char_data['glass_thicknesses'])
            analysis['glazing_specs_found'] = len(set(glazing_data))
            analysis['coverage'] = 'comprehensive' if len(set(glazing_data)) >= 2 else 'partial'
        
        elif self.characteristic == WindowCharacteristic.IMPACT_RATING:
            # Check for complete impact rating information
            impact_types = set()
            for item in extracted_content:
                data_points = item.get('table_analysis', {}).get('data_points_found', [])
                impact_types.update(data_points)
            analysis['impact_ratings_found'] = len(impact_types)
            analysis['coverage'] = 'comprehensive' if len(impact_types) >= 1 else 'partial'
        
        elif self.characteristic == WindowCharacteristic.DESIGN_PRESSURE:
            # Analyze pressure data completeness
            pressure_data = []
            for item in extracted_content:
                char_data = item.get('table_analysis', {}).get('characteristic_specific_data', {})
                if 'pressure_values' in char_data:
                    pressure_data.extend(char_data['pressure_values'])
            analysis['pressure_values_found'] = len(set(pressure_data))
            analysis['coverage'] = 'comprehensive' if len(set(pressure_data)) >= 2 else 'partial'
        
        return analysis
    
    def _create_extraction_data(self, doc_id: str, source: str, content: List[Dict], 
                              processing_time: float, analysis: Dict) -> Dict:
        """Create extraction data structure"""
        return {
            'document_id': doc_id,
            'document_path': source,
            'characteristic_focus': self.characteristic.value,
            'extracted_sections': content,
            'processing_time': processing_time,
            'parameters_used': self.params.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'total_sections': len(content),
            'extraction_analysis': analysis,
            'characteristic_config': self.characteristic_config
        }
    
    def _run_llm_feedback(self, doc_id: str, debug: bool = False) -> bool:
        """Run LLM feedback for characteristic-specific extraction"""
        try:
            required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                print(f"⚠️  Missing Azure OpenAI variables: {missing_vars}")
                return self._run_fallback_feedback(doc_id)
            
            print("🤖 Launching characteristic-specific LLM feedback analyzer...")
            
            cmd = [sys.executable, "llm_feedback.py", "--analyze-characteristic", 
                   self.characteristic.value, doc_id]
            if debug:
                cmd.append("--debug")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print("✅ LLM feedback completed successfully")
                
                # Check for parameter changes
                old_params = self.params.to_dict()
                self.params = self._load_characteristic_parameters()
                new_params = self.params.to_dict()
                
                changes = {k: v for k, v in new_params.items() if old_params.get(k) != v}
                if changes:
                    print(f"✓ {self.characteristic.value} parameters updated:")
                    for param, new_val in changes.items():
                        old_val = old_params.get(param, 'N/A')
                        print(f"    {param}: {old_val} → {new_val}")
                else:
                    print("ℹ️  No parameter changes recommended")
                
                return True
            else:
                print(f"⚠️  LLM feedback failed - using fallback")
                if debug and result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                return self._run_fallback_feedback(doc_id)
                
        except subprocess.TimeoutExpired:
            print("⏰ LLM feedback timed out - using fallback")
            return self._run_fallback_feedback(doc_id)
        except Exception as e:
            if debug:
                print(f"❌ LLM feedback error: {e}")
            return self._run_fallback_feedback(doc_id)
    
    def _run_fallback_feedback(self, doc_id: str) -> bool:
        """Run fallback feedback analysis for characteristic"""
        try:
            extraction_file = f"feedback_data/{self.characteristic.value}_extraction_{doc_id}.json"
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            sections = extraction_data.get('extracted_sections', [])
            analysis = extraction_data.get('extraction_analysis', {})
            current_params = extraction_data.get('parameters_used', {})
            
            print(f"🔧 Fallback analysis: {len(sections)} items extracted")
            
            adjustments = {}
            reasoning_parts = []
            
            # Characteristic-specific fallback logic
            avg_confidence = analysis.get('avg_confidence', 0)
            coverage = analysis.get('coverage', 'unknown')
            
            if len(sections) < 3:
                print(f"   ⚠ Very low extraction for {self.characteristic.value} - reducing thresholds")
                adjustments.update({
                    'confidence_threshold': 0.25,
                    'content_classification_threshold': 0.2,
                    'image_size_min': max(current_params.get('image_size_min', 150) - 30, 100)
                })
                reasoning_parts.append(f"Very low {self.characteristic.value} extraction - aggressive threshold reduction")
            
            elif len(sections) < 8:
                print(f"   ⚠ Low extraction for {self.characteristic.value} - minor adjustments")
                adjustments.update({
                    'confidence_threshold': max(current_params.get('confidence_threshold', 0.4) * 0.85, 0.3),
                    'content_classification_threshold': max(current_params.get('content_classification_threshold', 0.3) * 0.9, 0.25)
                })
                reasoning_parts.append(f"Low {self.characteristic.value} extraction - moderate threshold reduction")
            
            elif avg_confidence < 0.5:
                print(f"   ⚠ Low confidence for {self.characteristic.value} - adjusting classification")
                adjustments.update({
                    'content_classification_threshold': max(current_params.get('content_classification_threshold', 0.3) * 0.8, 0.2)
                })
                reasoning_parts.append(f"Low confidence in {self.characteristic.value} extraction - improving sensitivity")
            
            elif coverage == 'partial':
                print(f"   📊 Partial coverage for {self.characteristic.value} - expanding search")
                adjustments.update({
                    'max_extractions': min(current_params.get('max_extractions', 20) + 5, 30),
                    'table_relevance_threshold': max(current_params.get('table_relevance_threshold', 2) - 1, 1)
                })
                reasoning_parts.append(f"Partial {self.characteristic.value} coverage - expanding extraction scope")
            
            # Apply adjustments
            if adjustments:
                self._apply_parameter_changes(adjustments)
                print(f"   ✓ Applied {len(adjustments)} parameter adjustments")
            else:
                print(f"   ℹ️  No adjustments needed for {self.characteristic.value}")
            
            # Log fallback analysis
            self._save_fallback_log(doc_id, adjustments, reasoning_parts)
            return True
            
        except Exception as e:
            print(f"❌ Fallback analysis failed: {e}")
            return False
    
    def _apply_parameter_changes(self, adjustments: Dict):
        """Apply parameter changes with validation"""
        param_file = f"parameters_{self.characteristic.value}.json"
        
        try:
            if os.path.exists(param_file):
                with open(param_file) as f:
                    current_params = json.load(f)
            else:
                current_params = self.params.to_dict()
        except:
            current_params = self.params.to_dict()
        
        # Apply changes with validation
        changes_made = []
        for param_name, new_value in adjustments.items():
            if param_name in current_params:
                try:
                    # Type conversion and bounds checking
                    if param_name in ['max_extractions', 'image_size_min', 'cv_detection_threshold', 
                                    'min_section_length', 'table_relevance_threshold']:
                        new_value = max(1, int(float(new_value)))
                    else:
                        new_value = max(0.1, min(1.0, float(new_value)))
                    
                    old_value = current_params[param_name]
                    if old_value != new_value:
                        current_params[param_name] = new_value
                        changes_made.append(f"{param_name}: {old_value} → {new_value}")
                        print(f"      ✓ {param_name}: {old_value} → {new_value}")
                        
                except (ValueError, TypeError):
                    print(f"      ✗ Invalid value for {param_name}: {new_value}")
        
        # Save updated parameters
        if changes_made:
            with open(param_file, 'w') as f:
                json.dump(current_params, f, indent=2)
            self.params = CharacteristicParameters.from_dict(current_params)
    
    def _save_fallback_log(self, document_id: str, adjustments: Dict, reasoning_parts: List[str]):
        """Save fallback analysis log"""
        log_entry = {
            'document_id': document_id,
            'characteristic': self.characteristic.value,
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': f'Fallback analysis - {self.characteristic.value} extraction',
            'parameter_adjustments': adjustments,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'No adjustments needed',
            'llm_available': False,
            'llm_actually_used': False,
            'fallback_reason': f'{self.characteristic.value} characteristic fallback analysis'
        }
        
        log_file = f"feedback_log_{self.characteristic.value}.json"
        try:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"⚠ Error saving fallback log: {e}")
    
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
    
    def _extract_page_context(self, all_elements, page_num: int) -> Dict:
        """Extract context from specific page"""
        page_text = []
        for element, _ in all_elements:
            if self._get_page_number(element) == page_num and isinstance(element, TextItem):
                text_content = element.text if hasattr(element, 'text') else str(element)
                page_text.append(text_content)
        
        combined_text = ' '.join(page_text).lower()
        
        return {
            'page_text': combined_text,
            'text_length': len(combined_text),
            'page_number': page_num
        }
    
    def _download(self, url: str) -> str:
        """Download PDF from URL"""
        import requests
        print(f"🌐 Downloading from: {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        print(f"✓ Downloaded to temporary file")
        return tmp

def main():
    parser = argparse.ArgumentParser(description="Specialized Window Characteristics Extraction Agent")
    parser.add_argument("--source", required=True, help="PDF path or URL to process")
    parser.add_argument("--characteristic", required=True, 
                       choices=['anchors', 'glazing', 'impact_rating', 'design_pressure'],
                       help="Window characteristic to focus on")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--test-params", action="store_true", help="Show current parameters")
    
    args = parser.parse_args()
    
    try:
        characteristic = WindowCharacteristic(args.characteristic)
    except ValueError:
        print(f"❌ Invalid characteristic: {args.characteristic}")
        print("Valid options: anchors, glazing, impact_rating, design_pressure")
        return 1
    
    if args.test_params:
        agent = WindowCharacteristicAgent(characteristic)
        print(f"\n📊 CURRENT {characteristic.value.upper()} PARAMETERS")
        print("="*50)
        for param, value in agent.params.to_dict().items():
            print(f"{param:30}: {value}")
        print("="*50)
        return 0
    
    print(f"🎯 Starting {characteristic.value.title()} Extraction Agent...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    agent = WindowCharacteristicAgent(characteristic)
    
    if args.debug:
        print(f"\n🔧 DEBUG MODE ENABLED")
        print(f"Characteristic Focus: {characteristic.value}")
        print(f"Computer Vision Available: {CV_AVAILABLE}")
        print(f"Docling Available: {DOCLING_OK}")
        print("="*70)
    
    try:
        doc_id = agent.process_document(args.source, debug=args.debug)
        
        print(f"\n🎉 {characteristic.value.upper()} EXTRACTION SUCCESSFUL!")
        print(f"Document ID: {doc_id}")
        print(f"Focus: {characteristic.value.replace('_', ' ').title()}")
        print(f"Next steps:")
        print(f"  1. View results: streamlit run feedback_interface.py")
        print(f"  2. Test LLM: python llm_feedback.py --test-connection")
        print(f"  3. Check logs: python llm_feedback.py --show-log --characteristic {characteristic.value}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  {characteristic.value.title()} extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ {characteristic.value.upper()} EXTRACTION FAILED: {e}")
        if args.debug:
            import traceback
            print("\n🔍 Full error traceback:")
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
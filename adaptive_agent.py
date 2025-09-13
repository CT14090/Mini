#!/usr/bin/env python3
"""
Window Characteristics Extraction Agent - Clean Version
Extracts specific window characteristics with reference data and LLM feedback
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
from datetime import datetime
from typing import Dict, List, Optional
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
    print("Computer Vision available")
except ImportError:
    print("OpenCV/PIL not available")

# Docling imports
DOCLING_OK = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
    DOCLING_OK = True
    print("Docling available")
except Exception as e:
    print(f"Docling not available: {e}")

# LLM imports
LLM_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
    print("Azure OpenAI available")
except ImportError:
    print("Azure OpenAI not available")

class WindowCharacteristic(Enum):
    ANCHORS = "anchors"
    GLAZING = "glazing"
    IMPACT_RATING = "impact_rating"
    DESIGN_PRESSURE = "design_pressure"

@dataclass
class Parameters:
    confidence_threshold: float = 0.25
    min_section_length: int = 100
    max_extractions: int = 15
    image_size_min: int = 100
    skip_pages: int = 3
    content_classification_threshold: float = 0.15
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class WindowAgent:
    def __init__(self, characteristic: WindowCharacteristic):
        self.characteristic = characteristic
        print(f"Initializing {characteristic.value.title()} Agent")
        
        # Load parameters
        self.params = self._load_parameters()
        
        # Check dependencies
        if not DOCLING_OK:
            raise RuntimeError("Docling required. Install: pip install docling>=1.0.0")
        
        # Initialize document converter
        pp = PdfPipelineOptions()
        pp.images_scale = 2.8
        pp.generate_page_images = True
        pp.generate_picture_images = True
        pp.generate_table_images = True
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pp)}
        )
        
        # Load reference data and config
        self.config = self._get_config()
        self.reference_data = self._load_reference_data()
        
        # Check Azure OpenAI
        self.azure_available = self._check_azure()
        
        print(f"Agent ready - Azure: {self.azure_available}, References: {len(self.reference_data)}")
    
    def _load_parameters(self) -> Parameters:
        param_file = f"parameters_{self.characteristic.value}.json"
        if os.path.exists(param_file):
            try:
                with open(param_file) as f:
                    return Parameters.from_dict(json.load(f))
            except:
                pass
        
        # Create default
        params = Parameters()
        self._save_parameters(params)
        return params
    
    def _save_parameters(self, params: Parameters = None):
        if params is None:
            params = self.params
        with open(f"parameters_{self.characteristic.value}.json", 'w') as f:
            json.dump(params.to_dict(), f, indent=2)
    
    def _get_config(self) -> Dict:
        configs = {
            WindowCharacteristic.ANCHORS: {
                'keywords': [
                    'anchor', 'anchors', 'fastener', 'fasteners', 'screw', 'bolt',
                    'concrete screw', 'wood screw', 'structural screw', 
                    'attachment', 'connection', 'mounting', 'structural connection',
                    'buck', 'blocking', 'jamb anchor', 'frame attachment'
                ],
                'table_indicators': ['anchor', 'fastener', 'screw', 'attachment'],
                'image_patterns': ['anchor', 'screw', 'fastener', 'connection']
            },
            WindowCharacteristic.GLAZING: {
                'keywords': [
                    'glazing', 'glass', 'igu', 'insulated glass', 'low-e', 'low e',
                    'laminated', 'tempered', 'thickness', 'double glazed', 'triple glazed'
                ],
                'table_indicators': ['glass', 'glazing', 'thickness', 'specification'],
                'image_patterns': ['glass', 'glazing', 'section']
            },
            WindowCharacteristic.IMPACT_RATING: {
                'keywords': [
                    'impact', 'missile', 'hurricane', 'storm', 'impact rating',
                    'small missile', 'large missile', 'astm', 'aama'
                ],
                'table_indicators': ['impact', 'missile', 'rating', 'test'],
                'image_patterns': ['impact', 'test', 'rating']
            },
            WindowCharacteristic.DESIGN_PRESSURE: {
                'keywords': [
                    'design pressure', 'dp', 'pressure rating', 'wind load',
                    'psf', 'pa', 'kpa', 'structural load', 'wind rating'
                ],
                'table_indicators': ['pressure', 'dp', 'load', 'psf'],
                'image_patterns': ['pressure', 'load', 'table']
            }
        }
        return configs.get(self.characteristic, configs[WindowCharacteristic.ANCHORS])
    
    def _load_reference_data(self) -> List[str]:
        reference_dir = pathlib.Path("labeled_data") / self.characteristic.value
        if not reference_dir.exists():
            return []
        
        references = []
        for img_file in reference_dir.glob("*.{jpg,jpeg,png}"):
            references.append(str(img_file))
        
        return references
    
    def _check_azure(self) -> bool:
        required = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        return all(os.getenv(var) for var in required) and LLM_AVAILABLE
    
    def process_document(self, source: str, debug: bool = False) -> str:
        start_time = time.time()
        doc_id = hashlib.md5(f"{source}_{self.characteristic.value}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        print(f"\n=== {self.characteristic.value.upper()} EXTRACTION ===")
        print(f"Source: {os.path.basename(source)}")
        print(f"Document ID: {doc_id}")
        print(f"Skip pages: {self.params.skip_pages}")
        print(f"References: {len(self.reference_data)}")
        
        # Convert document
        print("Converting document...")
        conv_res = self.converter.convert(source)
        doc = conv_res.document
        all_elements = list(doc.iterate_items())
        print(f"Found {len(all_elements)} elements")
        
        # Filter by page (skip first N pages)
        filtered_elements = []
        for element, _ in all_elements:
            page = self._get_page_number(element)
            if page > self.params.skip_pages:
                filtered_elements.append((element, _))
        
        print(f"Processing {len(filtered_elements)} elements (skipped first {self.params.skip_pages} pages)")
        
        # Extract content
        extracted_content = []
        
        # Text summary
        text_summary = self._extract_text_summary(filtered_elements, debug)
        if text_summary['summary']:
            print(f"Generated text summary: {len(text_summary['summary'])} chars")
        
        # Images
        images = self._extract_images(doc, filtered_elements, text_summary, debug)
        print(f"Found {len(images)} relevant images")
        extracted_content.extend(images)
        
        # Tables
        tables = self._extract_tables(filtered_elements, debug)
        print(f"Found {len(tables)} relevant tables")
        extracted_content.extend(tables)
        
        # Add text summary if substantial
        if len(text_summary.get('summary', '')) > 200:
            extracted_content.append({
                'type': f'{self.characteristic.value}_summary',
                'content': text_summary['summary'][:1000],
                'page': 'multiple',
                'confidence': 0.8,
                'metadata': {'source_pages': text_summary.get('source_pages', [])}
            })
        
        # Sort and limit
        extracted_content.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        if len(extracted_content) > self.params.max_extractions:
            extracted_content = extracted_content[:self.params.max_extractions]
        
        processing_time = time.time() - start_time
        
        print(f"\nEXTRACTION SUMMARY:")
        print(f"Total items: {len(extracted_content)}")
        print(f"Processing time: {processing_time:.1f}s")
        
        # Save extraction data
        extraction_data = {
            'document_id': doc_id,
            'document_path': source,
            'characteristic_focus': self.characteristic.value,
            'extracted_sections': extracted_content,
            'text_summary': text_summary,
            'processing_time': processing_time,
            'parameters_used': self.params.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'reference_data_available': len(self.reference_data) > 0
        }
        
        # Save to file
        pathlib.Path("feedback_data").mkdir(exist_ok=True)
        extraction_file = f"feedback_data/{self.characteristic.value}_extraction_{doc_id}.json"
        with open(extraction_file, 'w') as f:
            json.dump(extraction_data, f, indent=2)
        
        print(f"Saved: {extraction_file}")
        
        # Run LLM feedback
        if self.azure_available:
            print("\nRunning LLM feedback...")
            success = self._run_llm_feedback(doc_id, source, debug)
            if success:
                print("LLM feedback completed")
            else:
                print("LLM feedback failed")
        else:
            print("LLM feedback not available (Azure not configured)")
        
        print(f"\nEXTRACTION COMPLETE: {doc_id}")
        return doc_id
    
    def _extract_text_summary(self, all_elements, debug: bool = False) -> Dict:
        page_texts = {}
        config = self.config
        
        # Group text by page
        for element, _ in all_elements:
            if isinstance(element, TextItem):
                page = self._get_page_number(element)
                text = element.text if hasattr(element, 'text') else str(element)
                if page not in page_texts:
                    page_texts[page] = []
                page_texts[page].append(text)
        
        # Find relevant sentences
        relevant_sentences = []
        for page, texts in page_texts.items():
            combined = ' '.join(texts)
            sentences = [s.strip() for s in combined.split('.') if len(s.strip()) > 30]
            
            for sentence in sentences:
                score = 0
                sentence_lower = sentence.lower()
                
                # Check keywords
                for keyword in config['keywords']:
                    if keyword.lower() in sentence_lower:
                        score += 0.1
                
                if score >= 0.1:
                    relevant_sentences.append({
                        'sentence': sentence,
                        'page': page,
                        'score': score
                    })
        
        # Create summary
        if relevant_sentences:
            relevant_sentences.sort(key=lambda x: x['score'], reverse=True)
            top_sentences = relevant_sentences[:8]
            
            summary_parts = []
            source_pages = []
            for sent in top_sentences:
                summary_parts.append(f"[Page {sent['page']}] {sent['sentence']}")
                if sent['page'] not in source_pages:
                    source_pages.append(sent['page'])
            
            return {
                'summary': '\n\n'.join(summary_parts),
                'source_pages': source_pages,
                'sentence_count': len(top_sentences)
            }
        
        return {'summary': '', 'source_pages': [], 'sentence_count': 0}
    
    def _extract_images(self, doc, all_elements, text_summary: Dict, debug: bool = False) -> List[Dict]:
        images = []
        picture_items = [element for element, _ in all_elements if isinstance(element, PictureItem)]
        
        for i, element in enumerate(picture_items):
            try:
                pil_image = element.get_image(doc)
                if not pil_image:
                    continue
                
                width, height = pil_image.size
                page = self._get_page_number(element)
                area = width * height
                
                if debug:
                    print(f"  Image {i+1}: {width}x{height} page {page}")
                
                # Size check
                if area < (self.params.image_size_min ** 2):
                    if debug:
                        print(f"    Too small: {area}")
                    continue
                
                # Get page context
                page_context = self._get_page_context(all_elements, page)
                
                # Calculate relevance
                score = self._calculate_image_relevance(page_context, text_summary)
                
                # Reference matching
                if self.reference_data and CV_AVAILABLE:
                    ref_score = self._match_references(pil_image)
                    score += ref_score
                
                if debug:
                    print(f"    Relevance: {score:.3f}")
                
                if score >= self.params.content_classification_threshold:
                    images.append({
                        'type': f'{self.characteristic.value}_image',
                        'content': f"{self.characteristic.value} image from page {page}",
                        'page': page,
                        'confidence': min(score, 1.0),
                        'data_uri': self._pil_to_data_uri(pil_image),
                        'metadata': {
                            'width': width,
                            'height': height,
                            'area': area
                        }
                    })
                    if debug:
                        print(f"    ✓ Extracted")
                else:
                    if debug:
                        print(f"    ✗ Rejected")
            
            except Exception as e:
                if debug:
                    print(f"  Error processing image {i+1}: {e}")
        
        return images
    
    def _extract_tables(self, all_elements, debug: bool = False) -> List[Dict]:
        tables = []
        table_items = [element for element, _ in all_elements if isinstance(element, TableItem)]
        
        for i, element in enumerate(table_items):
            try:
                content = element.export_to_markdown()
                page = self._get_page_number(element)
                
                if len(content.strip()) < self.params.min_section_length:
                    continue
                
                if debug:
                    print(f"  Table {i+1}: {len(content)} chars page {page}")
                
                score = self._calculate_table_relevance(content)
                
                if debug:
                    print(f"    Relevance: {score:.3f}")
                
                if score >= self.params.content_classification_threshold:
                    tables.append({
                        'type': f'{self.characteristic.value}_table',
                        'content': content,
                        'page': page,
                        'confidence': min(score, 1.0),
                        'metadata': {
                            'row_count': content.count('\n'),
                            'content_length': len(content)
                        }
                    })
                    if debug:
                        print(f"    ✓ Extracted")
                else:
                    if debug:
                        print(f"    ✗ Rejected")
            
            except Exception as e:
                if debug:
                    print(f"  Error processing table {i+1}: {e}")
        
        return tables
    
    def _calculate_image_relevance(self, page_context: Dict, text_summary: Dict) -> float:
        page_text = page_context.get('page_text', '').lower()
        summary_text = text_summary.get('summary', '').lower()
        config = self.config
        
        score = 0.0
        
        # Check page context
        for keyword in config['keywords']:
            if keyword.lower() in page_text:
                score += 0.05
        
        # Check image patterns
        for pattern in config['image_patterns']:
            if pattern.lower() in page_text:
                score += 0.1
        
        # Check summary relevance
        for keyword in config['keywords'][:5]:
            if keyword.lower() in summary_text:
                score += 0.05
        
        return min(score, 0.7)  # Leave room for reference matching
    
    def _calculate_table_relevance(self, content: str) -> float:
        content_lower = content.lower()
        config = self.config
        score = 0.0
        
        # Check table indicators
        for indicator in config['table_indicators']:
            if indicator.lower() in content_lower:
                score += 0.15
        
        # Check keywords
        for keyword in config['keywords']:
            if keyword.lower() in content_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _match_references(self, target_image) -> float:
        if not CV_AVAILABLE or not self.reference_data:
            return 0.0
        
        try:
            target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
            target_gray = cv2.cvtColor(target_cv, cv2.COLOR_BGR2GRAY)
            
            orb = cv2.ORB_create()
            target_kp, target_desc = orb.detectAndCompute(target_gray, None)
            
            if target_desc is None:
                return 0.0
            
            best_score = 0.0
            
            for ref_path in self.reference_data:
                try:
                    ref_image = cv2.imread(ref_path)
                    if ref_image is None:
                        continue
                    
                    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
                    ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
                    
                    if ref_desc is None:
                        continue
                    
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(target_desc, ref_desc)
                    
                    if len(matches) > 10:
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = [m for m in matches[:20] if m.distance < 80]
                        match_score = len(good_matches) / min(len(target_kp), len(ref_kp))
                        best_score = max(best_score, match_score)
                
                except:
                    continue
            
            return min(best_score * 0.3, 0.3)
        
        except:
            return 0.0
    
    def _run_llm_feedback(self, doc_id: str, source_pdf: str, debug: bool = False) -> bool:
        if not self.azure_available:
            return False
        
        try:
            # Load extraction data
            extraction_file = f"feedback_data/{self.characteristic.value}_extraction_{doc_id}.json"
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            # Initialize LLM
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=1500
            )
            
            # Prepare analysis data
            sections = extraction_data.get('extracted_sections', [])
            summary = extraction_data.get('text_summary', {})
            
            analysis_text = f"""
EXTRACTION ANALYSIS FOR {self.characteristic.value.upper()}:

Document: {os.path.basename(source_pdf)}
Total items extracted: {len(sections)}
Text summary available: {bool(summary.get('summary'))}
Reference data available: {len(self.reference_data) > 0}

EXTRACTED CONTENT:
"""
            
            for i, section in enumerate(sections[:5], 1):
                section_type = section.get('type', 'unknown')
                confidence = section.get('confidence', 0)
                page = section.get('page', 'unknown')
                content_preview = section.get('content', '')[:100] + "..."
                
                analysis_text += f"""
{i}. Type: {section_type}
   Page: {page}, Confidence: {confidence:.3f}
   Content: {content_preview}
"""
            
            # Create prompt
            system_prompt = f"""You are analyzing window {self.characteristic.value} extraction results. 

For {self.characteristic.value}, you should expect to find:
- Relevant technical specifications and details
- Tables with {self.characteristic.value}-specific data
- Images showing {self.characteristic.value}-related components
- Text describing {self.characteristic.value} installation or specifications

Evaluate the extraction quality and suggest parameter improvements."""
            
            user_prompt = f"""{analysis_text}

Current parameters:
{json.dumps(extraction_data.get('parameters_used', {}), indent=2)}

Please analyze this extraction and respond in JSON format:
{{
    "quality_assessment": {{
        "relevance": <1-5>,
        "completeness": <1-5>,
        "accuracy": <1-5>
    }},
    "feedback": {{
        "correctly_identified": ["item1", "item2"],
        "incorrectly_classified": ["item3"],
        "missing_content": ["item4"]
    }},
    "parameter_recommendations": {{
        "content_classification_threshold": <new_value>,
        "confidence_threshold": <new_value>,
        "reasoning": "explanation"
    }}
}}"""
            
            # Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            
            # Parse response
            try:
                response_content = response.content.strip()
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                
                analysis_result = json.loads(response_content)
                
                # Apply recommendations
                recommendations = analysis_result.get('parameter_recommendations', {})
                if recommendations:
                    changes_made = []
                    current_params = self.params.to_dict()
                    
                    for param, new_value in recommendations.items():
                        if param == 'reasoning':
                            continue
                        if param in current_params and new_value is not None:
                            try:
                                if param in ['content_classification_threshold', 'confidence_threshold']:
                                    new_value = max(0.05, min(0.95, float(new_value)))
                                else:
                                    new_value = float(new_value)
                                
                                old_value = current_params[param]
                                if abs(old_value - new_value) >= 0.01:
                                    current_params[param] = new_value
                                    changes_made.append(f"{param}: {old_value} → {new_value}")
                            except:
                                continue
                    
                    if changes_made:
                        self.params = Parameters.from_dict(current_params)
                        self._save_parameters()
                        print(f"Updated parameters: {changes_made}")
                    
                    # Save feedback log
                    self._save_feedback_log(doc_id, analysis_result, changes_made)
                
                return True
            
            except json.JSONDecodeError as e:
                if debug:
                    print(f"LLM response parsing error: {e}")
                return False
        
        except Exception as e:
            if debug:
                print(f"LLM feedback error: {e}")
            return False
    
    def _save_feedback_log(self, doc_id: str, analysis_result: Dict, changes_made: List[str]):
        log_entry = {
            'document_id': doc_id,
            'characteristic': self.characteristic.value,
            'timestamp': datetime.now().isoformat(),
            'quality_scores': analysis_result.get('quality_assessment', {}),
            'feedback': analysis_result.get('feedback', {}),
            'parameter_changes': changes_made,
            'reasoning': analysis_result.get('parameter_recommendations', {}).get('reasoning', '')
        }
        
        log_file = f"feedback_log_{self.characteristic.value}.json"
        logs = []
        
        if os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    logs = json.load(f)
            except:
                pass
        
        logs.append(log_entry)
        if len(logs) > 20:
            logs = logs[-20:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _get_page_number(self, element) -> int:
        try:
            if hasattr(element, 'prov') and element.prov and len(element.prov) > 0:
                return int(element.prov[0].page_no)
        except:
            pass
        return 1
    
    def _get_page_context(self, all_elements, page_num: int) -> Dict:
        page_text = []
        for element, _ in all_elements:
            if self._get_page_number(element) == page_num and isinstance(element, TextItem):
                text = element.text if hasattr(element, 'text') else str(element)
                page_text.append(text)
        
        return {
            'page_text': ' '.join(page_text).lower(),
            'page_number': page_num
        }
    
    def _pil_to_data_uri(self, pil_image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

def main():
    parser = argparse.ArgumentParser(description="Window Characteristics Extraction Agent")
    parser.add_argument("--source", required=True, help="PDF path or URL")
    parser.add_argument("--characteristic", required=True,
                       choices=['anchors', 'glazing', 'impact_rating', 'design_pressure'],
                       help="Window characteristic to extract")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--test-params", action="store_true", help="Show parameters")
    
    args = parser.parse_args()
    
    if not DOCLING_OK:
        print("ERROR: Docling not available. Install: pip install docling>=1.0.0")
        return 1
    
    try:
        characteristic = WindowCharacteristic(args.characteristic)
    except ValueError:
        print(f"Invalid characteristic: {args.characteristic}")
        return 1
    
    if args.test_params:
        agent = WindowAgent(characteristic)
        print(f"\nCURRENT {characteristic.value.upper()} PARAMETERS:")
        for param, value in agent.params.to_dict().items():
            print(f"  {param}: {value}")
        return 0
    
    try:
        agent = WindowAgent(characteristic)
        doc_id = agent.process_document(args.source, debug=args.debug)
        
        print(f"\nSUCCESS!")
        print(f"Document ID: {doc_id}")
        print(f"View results: streamlit run feedback_interface.py")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
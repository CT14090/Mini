#!/usr/bin/env python3
# llm_feedback.py
"""
Enhanced LLM Feedback System with Advanced Vision Analysis - VISUAL CONTENT FOCUSED
Provides intelligent feedback on visual content extraction quality
"""

import json
import os
import pathlib
import sys
import time
import base64
import argparse
import signal
from datetime import datetime
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError('LLM feedback timed out')

# LangChain imports
LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
    print("✓ LangChain available for Enhanced Azure OpenAI Vision")
except ImportError:
    print("⚠ LangChain not available")

class EnhancedVisionFeedbackAnalyzer:
    """Enhanced vision feedback with focus on visual content quality"""
    
    def __init__(self):
        self.azure_available = self._check_azure_config()
        self.llm = None
        self.analysis_timeout = 180  # 3 minutes max for enhanced analysis
        self.max_items_to_analyze = 8  # Increased for better analysis
        
        if self.azure_available and LANGCHAIN_AVAILABLE:
            try:
                self.llm = AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    api_version="2024-02-15-preview",
                    temperature=0.1,
                    max_tokens=1500  # Increased for detailed analysis
                )
                print("✓ Enhanced Azure OpenAI Vision configured")
            except Exception as e:
                print(f"⚠ Enhanced Azure OpenAI setup failed: {e}")
                self.azure_available = False
        
        # Load enhanced reference images
        self.labeled_data_path = pathlib.Path("labeled_data")
        self.reference_images = self._load_enhanced_reference_images()
    
    def _check_azure_config(self) -> bool:
        """Check Azure OpenAI configuration"""
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required_vars if not os.getenv(var)]
        return len(missing) == 0
    
    def _load_enhanced_reference_images(self) -> Dict[str, List[Dict]]:
        """Load enhanced reference images with metadata"""
        reference_images = {}
        
        if not self.labeled_data_path.exists():
            return reference_images
        
        print("📚 Loading enhanced reference images...")
        
        for category_dir in self.labeled_data_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                
                reference_images[category] = []
                
                # Load up to 3 reference images per category with metadata
                for img_file in image_files[:3]:
                    try:
                        with open(img_file, 'rb') as f:
                            img_data = f.read()
                            
                            # Skip very large images
                            if len(img_data) > 2000000:  # 2MB limit
                                continue
                                
                            b64_data = base64.b64encode(img_data).decode('ascii')
                            data_uri = f"data:image/jpeg;base64,{b64_data}"
                            
                            reference_images[category].append({
                                'data_uri': data_uri,
                                'filename': img_file.name,
                                'size_bytes': len(img_data),
                                'category': category
                            })
                    except Exception as e:
                        continue
                
                print(f"  ✓ {category}: {len(reference_images[category])} enhanced reference images")
        
        return reference_images
    
    def analyze_extraction_with_enhanced_vision(self, document_id: str, debug: bool = False) -> Dict:
        """Analyze extraction with enhanced vision capabilities"""
        # Set analysis timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.analysis_timeout)
        
        try:
            print(f"🔍 Enhanced vision analysis of extraction {document_id} (timeout: {self.analysis_timeout}s)...")
            
            # Load extraction data
            extraction_file = f"feedback_data/extraction_{document_id}.json"
            if not os.path.exists(extraction_file):
                raise FileNotFoundError(f"Extraction file not found: {extraction_file}")
            
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            extracted_sections = extraction_data.get('extracted_sections', [])
            current_params = extraction_data.get('parameters_used', {})
            target_characteristic = extraction_data.get('target_characteristic', 'unknown')
            
            # Enhanced item selection - prioritize by confidence and detection method
            items_to_analyze = self._select_items_for_analysis(extracted_sections, debug)
            print(f"  📊 Analyzing {len(items_to_analyze)} items (selected from {len(extracted_sections)} total)")
            
            if not self.azure_available or not self.llm:
                print("  ⚠ Azure OpenAI not available - using enhanced fallback")
                return self._enhanced_fallback_analysis(extraction_data, debug)
            
            # Enhanced vision analysis
            vision_results = []
            analysis_start_time = time.time()
            
            for i, section in enumerate(items_to_analyze):
                if time.time() - analysis_start_time > self.analysis_timeout * 0.8:
                    print(f"    ⚠ Approaching timeout, analyzed {i}/{len(items_to_analyze)} items")
                    break
                
                if 'data_uri' in section:
                    detection_method = section.get('region_metadata', {}).get('detection_method', 'unknown')
                    content_type = section.get('region_metadata', {}).get('content_type', 'unknown')
                    confidence = section.get('confidence', 0)
                    
                    print(f"    Analyzing item {i+1}: {content_type} via {detection_method} (confidence: {confidence:.3f})")
                    
                    try:
                        vision_result = self._analyze_single_item_enhanced(section, target_characteristic, debug)
                        vision_results.append(vision_result)
                    except Exception as e:
                        print(f"    ❌ Error analyzing item {i+1}: {e}")
                        continue
            
            # Generate enhanced analysis
            analysis_result = self._generate_enhanced_analysis(
                extraction_data, vision_results, current_params, debug
            )
            
            return analysis_result
            
        except TimeoutError:
            print(f"  ⚠ Enhanced analysis timed out after {self.analysis_timeout}s")
            return self._timeout_fallback_analysis(extraction_data if 'extraction_data' in locals() else {})
        
        except Exception as e:
            print(f"  ❌ Enhanced analysis failed: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return self._error_fallback_analysis(extraction_data if 'extraction_data' in locals() else {})
        
        finally:
            signal.alarm(0)  # Clear timeout
    
    def _select_items_for_analysis(self, sections: List[Dict], debug: bool) -> List[Dict]:
        """Intelligently select items for vision analysis"""
        if not sections:
            return []
        
        # Sort by multiple criteria
        def sort_key(item):
            confidence = item.get('confidence', 0)
            detection_method = item.get('region_metadata', {}).get('detection_method', 'unknown')
            content_type = item.get('region_metadata', {}).get('content_type', 'unknown')
            
            # Priority scoring
            priority_score = confidence
            
            # Boost specific detection methods
            if detection_method == 'table_detection':
                priority_score += 0.2
            elif detection_method == 'diagram_detection':
                priority_score += 0.15
            
            # Boost specific content types
            if content_type in ['table', 'diagram']:
                priority_score += 0.1
                
            return priority_score
        
        # Sort by priority and take top items
        sorted_sections = sorted(sections, key=sort_key, reverse=True)
        selected_items = sorted_sections[:self.max_items_to_analyze]
        
        if debug:
            print(f"    Selection criteria applied:")
            for i, item in enumerate(selected_items[:3]):  # Show top 3
                score = sort_key(item)
                method = item.get('region_metadata', {}).get('detection_method', 'unknown')
                content_type = item.get('region_metadata', {}).get('content_type', 'unknown')
                print(f"      {i+1}. Score: {score:.3f}, Method: {method}, Type: {content_type}")
        
        return selected_items
    
    def _analyze_single_item_enhanced(self, section: Dict, target_characteristic: str, debug: bool) -> Dict:
        """Enhanced analysis of single item with detailed vision prompts"""
        signal.alarm(45)  # 45 second timeout per item
        
        try:
            extracted_image = section.get('data_uri', '')
            predicted_category = section.get('type', 'unknown')
            region_metadata = section.get('region_metadata', {})
            detection_method = region_metadata.get('detection_method', 'unknown')
            content_type = region_metadata.get('content_type', 'unknown')
            
            if not extracted_image:
                return self._create_error_result('No image data available')
            
            # Get reference images for comparison
            reference_images = self.reference_images.get(predicted_category, [])
            
            # Create enhanced vision prompt
            messages = self._create_enhanced_vision_prompt(
                extracted_image, reference_images, predicted_category, 
                detection_method, content_type, target_characteristic
            )
            
            response = self.llm.invoke(messages)
            result = self._parse_enhanced_vision_response(response.content, section)
            
            return result
            
        except TimeoutError:
            return self._create_error_result('Analysis timed out')
        except Exception as e:
            return self._create_error_result(f'Error: {str(e)[:100]}')
        finally:
            signal.alarm(0)
    
    def _create_enhanced_vision_prompt(self, extracted_image: str, reference_images: List[Dict], 
                                     category: str, detection_method: str, content_type: str,
                                     target_characteristic: str) -> List[HumanMessage]:
        """Create enhanced vision prompt with detailed analysis instructions"""
        
        content = [
            {
                "type": "text",
                "text": f"""ENHANCED VISUAL CONTENT ANALYSIS

You are analyzing an extracted visual element for construction document processing.

EXTRACTION DETAILS:
- Target Characteristic: {target_characteristic.replace('_', ' ').title()}
- Predicted Category: {category.replace('_', ' ').title()}
- Detection Method: {detection_method.replace('_', ' ').title()}
- Content Type: {content_type.replace('_', ' ').title()}

EXTRACTED IMAGE TO ANALYZE:"""
            },
            {
                "type": "image_url",
                "image_url": {"url": extracted_image}
            }
        ]
        
        # Add reference images if available
        if reference_images:
            content.append({
                "type": "text", 
                "text": f"\nREFERENCE EXAMPLES for {category}:"
            })
            
            for i, ref_img in enumerate(reference_images[:2], 1):
                content.extend([
                    {
                        "type": "text",
                        "text": f"Reference {i} ({ref_img['filename']}):"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": ref_img['data_uri']}
                    }
                ])
        
        # Enhanced analysis instructions
        content.append({
            "type": "text",
            "text": f"""
ANALYSIS TASKS:

1. CONTENT VERIFICATION: Does the extracted image actually contain {target_characteristic.replace('_', ' ')} content?

2. VISUAL QUALITY: 
   - Is this a diagram, table, chart, or technical drawing?
   - Is it clearly readable and well-defined?
   - Does it contain structured visual information?

3. CATEGORY MATCH: Does this match the expected category "{category}"?

4. DETECTION METHOD ASSESSMENT: 
   - Method used: {detection_method}
   - Is this appropriate for the content type?

5. IMPROVEMENT SUGGESTIONS: How could extraction be improved?

Respond with JSON only:
{{
  "content_verification": {{
    "contains_target_characteristic": true/false,
    "confidence_level": "high/medium/low",
    "reasoning": "brief explanation"
  }},
  "visual_quality": {{
    "is_visual_content": true/false,
    "content_type": "table/diagram/chart/drawing/text/other",
    "readability": "excellent/good/fair/poor",
    "structured_information": true/false
  }},
  "category_match": {{
    "matches_expected_category": true/false,
    "actual_category_suggestion": "category name or 'unknown'",
    "similarity_to_references": "high/medium/low/no_references"
  }},
  "detection_method_assessment": {{
    "method_appropriate": true/false,
    "method_effectiveness": "excellent/good/fair/poor",
    "suggested_method": "method name or 'current_method_good'"
  }},
  "improvement_suggestions": [
    "suggestion 1",
    "suggestion 2"
  ],
  "overall_assessment": {{
    "extraction_quality": "excellent/good/fair/poor",
    "recommended_action": "keep/adjust_parameters/retrain/discard"
  }}
}}"""
        })
        
        return [HumanMessage(content=content)]
    
    def _parse_enhanced_vision_response(self, response_text: str, section: Dict) -> Dict:
        """Parse enhanced vision response with fallback handling"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
                
                # Add section metadata to response
                parsed_response['original_section_metadata'] = section.get('region_metadata', {})
                parsed_response['original_confidence'] = section.get('confidence', 0)
                
                return parsed_response
        except Exception as e:
            pass
        
        # Enhanced fallback parsing
        lower_response = response_text.lower()
        
        return {
            'content_verification': {
                'contains_target_characteristic': any(word in lower_response for word in ['yes', 'true', 'contains', 'matches']),
                'confidence_level': 'medium',
                'reasoning': 'Parsed from unstructured response'
            },
            'visual_quality': {
                'is_visual_content': any(word in lower_response for word in ['diagram', 'table', 'chart', 'drawing', 'visual']),
                'content_type': 'visual' if 'visual' in lower_response else 'unknown',
                'readability': 'fair',
                'structured_information': 'structure' in lower_response
            },
            'category_match': {
                'matches_expected_category': 'match' in lower_response,
                'actual_category_suggestion': 'unknown',
                'similarity_to_references': 'medium'
            },
            'detection_method_assessment': {
                'method_appropriate': True,
                'method_effectiveness': 'fair',
                'suggested_method': 'current_method_good'
            },
            'improvement_suggestions': ['Consider improving image quality', 'Review training data'],
            'overall_assessment': {
                'extraction_quality': 'fair',
                'recommended_action': 'keep'
            },
            'parsing_note': 'Fallback parsing used due to response format issues'
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            'content_verification': {
                'contains_target_characteristic': False,
                'confidence_level': 'low',
                'reasoning': error_message
            },
            'visual_quality': {
                'is_visual_content': False,
                'content_type': 'error',
                'readability': 'poor',
                'structured_information': False
            },
            'category_match': {
                'matches_expected_category': False,
                'actual_category_suggestion': 'unknown',
                'similarity_to_references': 'low'
            },
            'detection_method_assessment': {
                'method_appropriate': False,
                'method_effectiveness': 'poor',
                'suggested_method': 'unknown'
            },
            'improvement_suggestions': ['Resolve analysis error'],
            'overall_assessment': {
                'extraction_quality': 'poor',
                'recommended_action': 'discard'
            },
            'error_message': error_message
        }
    
    def _generate_enhanced_analysis(self, extraction_data: Dict, vision_results: List[Dict], 
                                  current_params: Dict, debug: bool) -> Dict:
        """Generate comprehensive enhanced analysis"""
        total_analyzed = len(vision_results)
        
        if total_analyzed == 0:
            return self._enhanced_fallback_analysis(extraction_data, debug)
        
        # Analyze vision results
        analysis_metrics = self._calculate_enhanced_metrics(vision_results)
        
        # Generate parameter recommendations
        param_recommendations = self._generate_enhanced_parameter_recommendations(
            extraction_data, analysis_metrics, current_params, debug
        )
        
        print(f"  📈 Enhanced Analysis Results:")
        print(f"    Accuracy: {analysis_metrics['accuracy_rate']:.1%}")
        print(f"    Visual content rate: {analysis_metrics['visual_content_rate']:.1%}")
        print(f"    Category match rate: {analysis_metrics['category_match_rate']:.1%}")
        
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'enhanced_vision_analysis': {
                'total_items_analyzed': total_analyzed,
                'accuracy_metrics': analysis_metrics,
                'visual_quality_assessment': self._assess_visual_quality(vision_results),
                'detection_method_effectiveness': self._assess_detection_methods(vision_results),
                'category_matching_performance': self._assess_category_matching(vision_results)
            },
            'parameter_recommendations': param_recommendations,
            'llm_available': True,
            'llm_actually_used': True,
            'analysis_method': 'enhanced_vision_analysis',
            'reference_images_used': sum(len(imgs) for imgs in self.reference_images.values()),
            'analysis_version': '2.0_enhanced'
        }
    
    def _calculate_fallback_metrics(self, sections: List[Dict]) -> Dict:
        """Calculate metrics using only metadata (no vision)"""
        if not sections:
            return {'total_items': 0}
        
        total = len(sections)
        
        # Analyze detection methods
        detection_methods = {}
        content_types = {}
        confidence_levels = []
        
        for section in sections:
            metadata = section.get('region_metadata', {})
            
            # Detection method analysis
            method = metadata.get('detection_method', 'unknown')
            detection_methods[method] = detection_methods.get(method, 0) + 1
            
            # Content type analysis
            content_type = metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Confidence analysis
            confidence = section.get('confidence', 0)
            confidence_levels.append(confidence)
        
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        high_confidence_count = sum(1 for c in confidence_levels if c > 0.6)
        
        # Estimate quality based on detection methods and content types
        visual_content_estimate = (
            content_types.get('table', 0) + 
            content_types.get('diagram', 0)
        ) / total if total > 0 else 0
        
        return {
            'total_items': total,
            'detection_method_breakdown': detection_methods,
            'content_type_breakdown': content_types,
            'average_confidence': avg_confidence,
            'high_confidence_items': high_confidence_count,
            'estimated_visual_content_rate': visual_content_estimate,
            'confidence_distribution': {
                'high': sum(1 for c in confidence_levels if c > 0.6),
                'medium': sum(1 for c in confidence_levels if 0.4 <= c <= 0.6),
                'low': sum(1 for c in confidence_levels if c < 0.4)
            }
        }
    
    def _fallback_parameter_recommendations(self, metrics: Dict, current_params: Dict) -> Dict:
        """Enhanced fallback parameter recommendations"""
        recommendations = {}
        reasoning = []
        
        total = metrics.get('total_items', 0)
        avg_confidence = metrics.get('average_confidence', 0)
        visual_content_rate = metrics.get('estimated_visual_content_rate', 0)
        high_conf_count = metrics.get('high_confidence_items', 0)
        
        if total == 0:
            recommendations['confidence_threshold'] = 0.4
            recommendations['min_region_size'] = 8000
            reasoning.append("No extractions found - reducing barriers significantly")
        
        elif total < 3:
            if avg_confidence > 0.6:
                recommendations['confidence_threshold'] = 0.5
                reasoning.append(f"Few but high-quality extractions ({total}) - slightly reducing threshold")
            else:
                recommendations['confidence_threshold'] = 0.45
                recommendations['min_region_size'] = 8000
                reasoning.append(f"Very few extractions ({total}) with low confidence - reducing barriers")
        
        elif total > 25:
            recommendations['confidence_threshold'] = 0.65
            recommendations['min_region_size'] = 12000
            reasoning.append(f"High extraction count ({total}) - increasing selectivity")
        
        elif visual_content_rate < 0.3:
            recommendations['confidence_threshold'] = 0.6
            reasoning.append(f"Low estimated visual content rate ({visual_content_rate:.1%}) - being more selective")
        
        elif high_conf_count / total < 0.3 if total > 0 else False:
            recommendations['confidence_threshold'] = 0.55
            reasoning.append(f"Low proportion of high-confidence items - moderate increase in threshold")
        
        return {
            'adjustments': recommendations,
            'reasoning': '; '.join(reasoning) if reasoning else 'Current parameters appear reasonable based on metadata analysis',
            'fallback_analysis_based_on': {
                'total_items': total,
                'average_confidence': avg_confidence,
                'estimated_visual_content_rate': visual_content_rate,
                'high_confidence_proportion': high_conf_count / total if total > 0 else 0
            }
        }
    
    def _timeout_fallback_analysis(self, extraction_data: Dict) -> Dict:
        """Analysis when timeout occurs"""
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'timeout_analysis': True,
            'parameter_recommendations': {
                'adjustments': {'confidence_threshold': 0.6},
                'reasoning': 'Enhanced analysis timed out - using safe defaults'
            },
            'llm_available': True,
            'llm_actually_used': False,
            'analysis_method': 'timeout_fallback',
            'analysis_version': '2.0_timeout'
        }
    
    def _error_fallback_analysis(self, extraction_data: Dict) -> Dict:
        """Analysis when error occurs"""
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'error_analysis': True,
            'parameter_recommendations': {
                'adjustments': {},
                'reasoning': 'Enhanced analysis failed - no changes recommended'
            },
            'llm_available': False,
            'llm_actually_used': False,
            'analysis_method': 'error_fallback',
            'analysis_version': '2.0_error'
        }
    
    def apply_enhanced_recommendations(self, analysis_result: Dict, debug: bool = False) -> bool:
        """Apply enhanced parameter recommendations safely"""
        recommendations = analysis_result.get('parameter_recommendations', {})
        adjustments = recommendations.get('adjustments', {})
        
        if not adjustments:
            print("  ℹ️ No parameter adjustments recommended by enhanced analysis")
            return True
        
        print(f"  ⚙️ Applying {len(adjustments)} enhanced parameter adjustments...")
        
        # Load current parameters
        param_file = "learning_parameters.json"
        try:
            if os.path.exists(param_file):
                with open(param_file) as f:
                    current_params = json.load(f)
            else:
                current_params = {
                    'confidence_threshold': 0.5,
                    'min_region_size': 10000,
                    'similarity_threshold': 0.6
                }
        except Exception as e:
            print(f"    ⚠ Error loading parameters: {e}")
            return False
        
        # Apply enhanced adjustments with improved bounds
        changes_made = []
        for param_name, new_value in adjustments.items():
            try:
                # Enhanced validation bounds
                if param_name == 'confidence_threshold':
                    new_value = max(0.25, min(0.85, float(new_value)))  # Wider range
                elif param_name == 'min_region_size':
                    new_value = max(5000, min(30000, int(new_value)))   # Wider range
                elif param_name == 'similarity_threshold':
                    new_value = max(0.3, min(0.9, float(new_value)))    # Wider range
                
                old_value = current_params.get(param_name, 'N/A')
                if old_value != new_value:
                    current_params[param_name] = new_value
                    changes_made.append(f"{param_name}: {old_value} → {new_value}")
                    
                    if debug:
                        print(f"      ✓ {param_name}: {old_value} → {new_value}")
                        
            except Exception as e:
                print(f"      ✗ Invalid value for {param_name}: {e}")
        
        # Save parameters with enhanced metadata
        if changes_made:
            try:
                # Add metadata about the enhancement
                current_params['_metadata'] = {
                    'last_updated': datetime.now().isoformat(),
                    'updated_by': 'enhanced_vision_feedback',
                    'analysis_version': analysis_result.get('analysis_version', '2.0'),
                    'changes_applied': len(changes_made)
                }
                
                with open(param_file, 'w') as f:
                    json.dump(current_params, f, indent=2)
                print(f"    ✅ Applied {len(changes_made)} enhanced parameter changes")
                
                # Show reasoning if available
                reasoning = recommendations.get('reasoning', '')
                if reasoning and debug:
                    print(f"    💡 Reasoning: {reasoning}")
                
                return True
            except Exception as e:
                print(f"    ❌ Error saving enhanced parameters: {e}")
                return False
        else:
            print("    ℹ️ No valid changes to apply from enhanced analysis")
            return True
    
    def save_enhanced_analysis_log(self, analysis_result: Dict):
        """Save enhanced analysis log with additional metadata"""
        log_file = "feedback_log.json"
        
        try:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add enhanced metadata
            analysis_result['log_metadata'] = {
                'logged_at': datetime.now().isoformat(),
                'log_version': '2.0_enhanced',
                'analysis_features_used': [
                    'enhanced_vision_prompts',
                    'multi_metric_analysis', 
                    'intelligent_item_selection',
                    'detection_method_assessment'
                ]
            }
            
            logs.append(analysis_result)
            
            # Keep only last 25 entries (increased for enhanced analysis)
            if len(logs) > 25:
                logs = logs[-25:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            print(f"  💾 Enhanced analysis saved to {log_file}")
            
        except Exception as e:
            print(f"  ⚠ Error saving enhanced analysis log: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced LLM Feedback with Advanced Vision Analysis")
    parser.add_argument("--analyze-and-apply", help="Analyze document ID and apply enhanced recommendations")
    parser.add_argument("--test-connection", action="store_true", help="Test Azure OpenAI connection")
    parser.add_argument("--show-log", action="store_true", help="Show recent enhanced feedback log")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--analyze-only", help="Run enhanced analysis without applying recommendations")
    
    args = parser.parse_args()
    
    if args.test_connection:
        print("🧪 Testing Enhanced Azure OpenAI connection...")
        analyzer = EnhancedVisionFeedbackAnalyzer()
        
        if analyzer.azure_available and analyzer.llm:
            try:
                messages = [HumanMessage(content="Test enhanced vision connection")]
                response = analyzer.llm.invoke(messages)
                print("✅ Enhanced Azure OpenAI connection successful")
                print(f"Response length: {len(response.content)} characters")
                return 0
            except Exception as e:
                print(f"❌ Enhanced connection test failed: {e}")
                return 1
        else:
            print("❌ Enhanced Azure OpenAI not properly configured")
            return 1
    
    if args.show_log:
        try:
            with open("feedback_log.json") as f:
                logs = json.load(f)
            
            print(f"\n📊 Enhanced Feedback Log ({len(logs)} entries)")
            print("="*70)
            
            for log in logs[-7:]:  # Show last 7
                timestamp = log.get('timestamp', 'Unknown')
                doc_id = log.get('document_id', 'Unknown')
                method = log.get('analysis_method', 'Unknown')
                version = log.get('analysis_version', 'Unknown')
                
                print(f"🕒 {timestamp}")
                print(f"📄 Document: {doc_id}")
                print(f"🔍 Method: {method} (v{version})")
                
                # Enhanced analysis results
                if 'enhanced_vision_analysis' in log:
                    eva = log['enhanced_vision_analysis']
                    metrics = eva.get('accuracy_metrics', {})
                    print(f"🎯 Accuracy: {metrics.get('accuracy_rate', 0):.1%}")
                    print(f"👁️ Visual Content: {metrics.get('visual_content_rate', 0):.1%}")
                    print(f"🎨 Category Match: {metrics.get('category_match_rate', 0):.1%}")
                    print(f"⭐ High Quality: {metrics.get('high_quality_rate', 0):.1%}")
                
                print("-" * 50)
            
            return 0
            
        except FileNotFoundError:
            print("📄 No enhanced feedback log found")
            return 0
        except Exception as e:
            print(f"❌ Error reading enhanced log: {e}")
            return 1
    
    if args.analyze_and_apply:
        document_id = args.analyze_and_apply
        print(f"🔍 Running enhanced vision analysis on document {document_id}...")
        
        try:
            analyzer = EnhancedVisionFeedbackAnalyzer()
            
            # Perform enhanced analysis
            analysis_result = analyzer.analyze_extraction_with_enhanced_vision(document_id, args.debug)
            
            # Apply enhanced recommendations
            success = analyzer.apply_enhanced_recommendations(analysis_result, args.debug)
            
            # Save enhanced analysis log
            analyzer.save_enhanced_analysis_log(analysis_result)
            
            if success:
                print("✅ Enhanced vision-based feedback analysis completed")
                return 0
            else:
                print("⚠️ Enhanced analysis completed with some issues")
                return 1
                
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    if args.analyze_only:
        document_id = args.analyze_only
        print(f"🔍 Running enhanced analysis only (no parameter changes) on document {document_id}...")
        
        try:
            analyzer = EnhancedVisionFeedbackAnalyzer()
            
            # Perform enhanced analysis only
            analysis_result = analyzer.analyze_extraction_with_enhanced_vision(document_id, args.debug)
            
            # Save analysis log but don't apply recommendations
            analyzer.save_enhanced_analysis_log(analysis_result)
            
            print("✅ Enhanced vision analysis completed (no parameters changed)")
            
            # Show key results
            if 'enhanced_vision_analysis' in analysis_result:
                eva = analysis_result['enhanced_vision_analysis']
                metrics = eva.get('accuracy_metrics', {})
                print(f"📊 Key Results:")
                print(f"  Accuracy: {metrics.get('accuracy_rate', 0):.1%}")
                print(f"  Visual Content Rate: {metrics.get('visual_content_rate', 0):.1%}")
                print(f"  Category Match Rate: {metrics.get('category_match_rate', 0):.1%}")
            
            return 0
                
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    print("❓ No action specified. Use --help for enhanced options.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
        """Calculate comprehensive metrics from vision analysis"""
        if not vision_results:
            return {}
        
        total = len(vision_results)
        
        # Content verification metrics
        contains_target = sum(1 for result in vision_results 
                            if result.get('content_verification', {}).get('contains_target_characteristic', False))
        
        # Visual quality metrics
        is_visual = sum(1 for result in vision_results
                       if result.get('visual_quality', {}).get('is_visual_content', False))
        
        good_readability = sum(1 for result in vision_results
                             if result.get('visual_quality', {}).get('readability', 'poor') in ['excellent', 'good'])
        
        # Category matching metrics
        category_matches = sum(1 for result in vision_results
                             if result.get('category_match', {}).get('matches_expected_category', False))
        
        # Quality assessment
        high_quality = sum(1 for result in vision_results
                          if result.get('overall_assessment', {}).get('extraction_quality', 'poor') in ['excellent', 'good'])
        
        return {
            'accuracy_rate': contains_target / total,
            'visual_content_rate': is_visual / total,
            'readability_rate': good_readability / total,
            'category_match_rate': category_matches / total,
            'high_quality_rate': high_quality / total,
            'total_analyzed': total
        }
    
    def _assess_visual_quality(self, vision_results: List[Dict]) -> Dict:
        """Assess overall visual quality of extractions"""
        content_types = {}
        readability_levels = {}
        
        for result in vision_results:
            vq = result.get('visual_quality', {})
            
            content_type = vq.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            readability = vq.get('readability', 'unknown')
            readability_levels[readability] = readability_levels.get(readability, 0) + 1
        
        return {
            'content_type_distribution': content_types,
            'readability_distribution': readability_levels,
            'dominant_content_type': max(content_types.items(), key=lambda x: x[1])[0] if content_types else 'unknown'
        }
    
    def _assess_detection_methods(self, vision_results: List[Dict]) -> Dict:
        """Assess effectiveness of detection methods"""
        method_effectiveness = {}
        
        for result in vision_results:
            dm = result.get('detection_method_assessment', {})
            original_method = result.get('original_section_metadata', {}).get('detection_method', 'unknown')
            
            effectiveness = dm.get('method_effectiveness', 'unknown')
            
            if original_method not in method_effectiveness:
                method_effectiveness[original_method] = []
            method_effectiveness[original_method].append(effectiveness)
        
        # Calculate average effectiveness per method
        method_scores = {}
        effectiveness_values = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 1}
        
        for method, scores in method_effectiveness.items():
            if scores:
                avg_score = sum(effectiveness_values.get(score, 1) for score in scores) / len(scores)
                method_scores[method] = {
                    'average_score': avg_score,
                    'sample_size': len(scores),
                    'effectiveness_rating': self._score_to_rating(avg_score)
                }
        
        return method_scores
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= 3.5:
            return 'excellent'
        elif score >= 2.5:
            return 'good'
        elif score >= 1.5:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_category_matching(self, vision_results: List[Dict]) -> Dict:
        """Assess category matching performance"""
        matching_performance = {
            'total_matches': 0,
            'total_items': len(vision_results),
            'suggested_categories': {}
        }
        
        for result in vision_results:
            cm = result.get('category_match', {})
            
            if cm.get('matches_expected_category', False):
                matching_performance['total_matches'] += 1
            
            suggested = cm.get('actual_category_suggestion', 'unknown')
            if suggested != 'unknown':
                matching_performance['suggested_categories'][suggested] = \
                    matching_performance['suggested_categories'].get(suggested, 0) + 1
        
        matching_performance['match_rate'] = (
            matching_performance['total_matches'] / matching_performance['total_items']
            if matching_performance['total_items'] > 0 else 0
        )
        
        return matching_performance
    
    def _generate_enhanced_parameter_recommendations(self, extraction_data: Dict, 
                                                   analysis_metrics: Dict, current_params: Dict, 
                                                   debug: bool) -> Dict:
        """Generate intelligent parameter recommendations based on enhanced analysis"""
        recommendations = {}
        reasoning = []
        
        accuracy_rate = analysis_metrics.get('accuracy_rate', 0)
        visual_content_rate = analysis_metrics.get('visual_content_rate', 0)
        category_match_rate = analysis_metrics.get('category_match_rate', 0)
        high_quality_rate = analysis_metrics.get('high_quality_rate', 0)
        
        total_extracted = extraction_data.get('total_sections', 0)
        
        # Enhanced logic based on multiple metrics
        if accuracy_rate < 0.4:
            # Low accuracy - increase selectivity
            recommendations['confidence_threshold'] = 0.7
            recommendations['min_region_size'] = 12000
            reasoning.append(f"Low accuracy ({accuracy_rate:.1%}) - increasing selectivity")
        
        elif visual_content_rate < 0.5:
            # Extracting too much non-visual content
            recommendations['confidence_threshold'] = 0.65
            reasoning.append(f"Low visual content rate ({visual_content_rate:.1%}) - focusing on visual elements")
        
        elif category_match_rate < 0.3:
            # Poor category matching
            recommendations['similarity_threshold'] = 0.7
            reasoning.append(f"Poor category matching ({category_match_rate:.1%}) - increasing similarity requirements")
        
        elif total_extracted > 30:
            # Too many extractions
            recommendations['confidence_threshold'] = 0.6
            recommendations['min_region_size'] = 15000
            reasoning.append(f"High extraction count ({total_extracted}) - increasing filters")
        
        elif total_extracted < 2 and accuracy_rate > 0.7:
            # Very few but high quality extractions - could be more permissive
            recommendations['confidence_threshold'] = 0.45
            recommendations['min_region_size'] = 8000
            reasoning.append(f"Few high-quality extractions ({total_extracted}) - slightly reducing barriers")
        
        elif high_quality_rate > 0.8 and total_extracted < 5:
            # High quality but few extractions
            recommendations['confidence_threshold'] = 0.5
            reasoning.append(f"High quality rate ({high_quality_rate:.1%}) but few items - reducing threshold")
        
        return {
            'adjustments': recommendations,
            'reasoning': '; '.join(reasoning) if reasoning else 'Analysis indicates current parameters are appropriate',
            'analysis_based_on': {
                'accuracy_rate': accuracy_rate,
                'visual_content_rate': visual_content_rate,
                'category_match_rate': category_match_rate,
                'high_quality_rate': high_quality_rate,
                'total_extracted': total_extracted
            }
        }
    
    def _enhanced_fallback_analysis(self, extraction_data: Dict, debug: bool) -> Dict:
        """Enhanced fallback without vision"""
        print("  🔧 Running enhanced fallback analysis...")
        
        sections = extraction_data.get('extracted_sections', [])
        current_params = extraction_data.get('parameters_used', {})
        
        # Enhanced heuristic analysis based on metadata
        fallback_metrics = self._calculate_fallback_metrics(sections)
        param_recommendations = self._fallback_parameter_recommendations(fallback_metrics, current_params)
        
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'enhanced_fallback_analysis': fallback_metrics,
            'parameter_recommendations': param_recommendations,
            'llm_available': False,
            'llm_actually_used': False,
            'analysis_method': 'enhanced_fallback_heuristic',
            'analysis_version': '2.0_fallback'
        }
    
    def _calculate_
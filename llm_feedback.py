#!/usr/bin/env python3
# llm_feedback.py
"""
Enhanced LLM Feedback System with Vision-Based Validation - FIXED VERSION
Timeout protection and simplified processing to prevent infinite loops
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
    print("✓ LangChain available for Azure OpenAI")
except ImportError:
    print("⚠ LangChain not available")

class VisionFeedbackAnalyzer:
    """Vision feedback with timeout protection"""
    
    def __init__(self):
        self.azure_available = self._check_azure_config()
        self.llm = None
        self.analysis_timeout = 120  # 2 minutes max for entire analysis
        self.max_items_to_analyze = 5  # Limit to prevent overuse
        
        if self.azure_available and LANGCHAIN_AVAILABLE:
            try:
                self.llm = AzureChatOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    api_version="2024-02-15-preview",
                    temperature=0.1,
                    max_tokens=1000  # Limit token usage
                )
                print("✓ Azure OpenAI Vision configured")
            except Exception as e:
                print(f"⚠ Azure OpenAI setup failed: {e}")
                self.azure_available = False
        
        # Load reference images (limited)
        self.labeled_data_path = pathlib.Path("labeled_data")
        self.reference_images = self._load_reference_images()
    
    def _check_azure_config(self) -> bool:
        """Check Azure OpenAI configuration"""
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required_vars if not os.getenv(var)]
        return len(missing) == 0
    
    def _load_reference_images(self) -> Dict[str, List[str]]:
        """Load limited reference images for comparison"""
        reference_images = {}
        
        if not self.labeled_data_path.exists():
            return reference_images
        
        print("📚 Loading reference images (limited)...")
        
        for category_dir in self.labeled_data_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                
                reference_images[category] = []
                # Limit to 2 reference images per category to prevent overuse
                for img_file in image_files[:2]:
                    try:
                        with open(img_file, 'rb') as f:
                            img_data = f.read()
                            # Limit image size
                            if len(img_data) > 1000000:  # 1MB limit
                                continue
                            b64_data = base64.b64encode(img_data).decode('ascii')
                            data_uri = f"data:image/jpeg;base64,{b64_data}"
                            reference_images[category].append(data_uri)
                    except Exception as e:
                        continue
                
                print(f"  ✓ {category}: {len(reference_images[category])} reference images")
        
        return reference_images
    
    def analyze_extraction_with_vision(self, document_id: str, debug: bool = False) -> Dict:
        """Analyze extraction with timeout protection"""
        # Set analysis timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.analysis_timeout)
        
        try:
            print(f"🔍 Analyzing extraction {document_id} (timeout: {self.analysis_timeout}s)...")
            
            # Load extraction data
            extraction_file = f"feedback_data/extraction_{document_id}.json"
            if not os.path.exists(extraction_file):
                raise FileNotFoundError(f"Extraction file not found: {extraction_file}")
            
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            extracted_sections = extraction_data.get('extracted_sections', [])
            current_params = extraction_data.get('parameters_used', {})
            
            # Limit items to analyze
            items_to_analyze = extracted_sections[:self.max_items_to_analyze]
            print(f"  📊 Analyzing {len(items_to_analyze)} items (limited from {len(extracted_sections)})")
            
            if not self.azure_available or not self.llm:
                print("  ⚠ Azure OpenAI not available - using fallback")
                return self._enhanced_fallback_analysis(extraction_data, debug)
            
            # Analyze with vision
            vision_results = []
            for i, section in enumerate(items_to_analyze):
                if 'data_uri' in section:
                    print(f"    Analyzing item {i+1}: {section.get('type', 'unknown')}")
                    
                    try:
                        vision_result = self._analyze_single_item_safe(section, debug)
                        vision_results.append(vision_result)
                    except Exception as e:
                        print(f"    ❌ Error analyzing item {i+1}: {e}")
                        continue
            
            # Generate analysis
            analysis_result = self._generate_analysis(
                extraction_data, vision_results, current_params, debug
            )
            
            return analysis_result
            
        except TimeoutError:
            print(f"  ⚠ Analysis timed out after {self.analysis_timeout}s")
            return self._timeout_fallback_analysis(extraction_data if 'extraction_data' in locals() else {})
        
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            return self._error_fallback_analysis(extraction_data if 'extraction_data' in locals() else {})
        
        finally:
            signal.alarm(0)  # Clear timeout
    
    def _analyze_single_item_safe(self, section: Dict, debug: bool) -> Dict:
        """Analyze single item with timeout protection"""
        signal.alarm(30)  # 30 second timeout per item
        
        try:
            extracted_image = section.get('data_uri', '')
            predicted_category = section.get('type', 'unknown')
            
            if not extracted_image or predicted_category not in self.reference_images:
                return {
                    'category_match': False,
                    'confidence_assessment': 'low',
                    'actual_content_type': 'unknown',
                    'reason': 'No reference images available'
                }
            
            # Get reference images
            reference_images = self.reference_images[predicted_category][:1]  # Use only 1 reference
            
            # Create simplified vision prompt
            messages = self._create_simple_vision_prompt(
                extracted_image, reference_images, predicted_category
            )
            
            response = self.llm.invoke(messages)
            result = self._parse_vision_response(response.content)
            
            return result
            
        except TimeoutError:
            return {
                'category_match': False,
                'confidence_assessment': 'timeout',
                'actual_content_type': 'unknown',
                'reason': 'Analysis timed out'
            }
        except Exception as e:
            return {
                'category_match': False,
                'confidence_assessment': 'error',
                'actual_content_type': 'unknown',
                'reason': f'Error: {str(e)[:50]}'
            }
        finally:
            signal.alarm(0)
    
    def _create_simple_vision_prompt(self, extracted_image: str, reference_images: List[str], category: str) -> List[HumanMessage]:
        """Create simplified vision prompt"""
        content = [
            {
                "type": "text",
                "text": f"""Compare these images. Does the EXTRACTED image match the REFERENCE for category "{category}"?

EXTRACTED IMAGE:"""
            },
            {
                "type": "image_url",
                "image_url": {"url": extracted_image}
            }
        ]
        
        if reference_images:
            content.extend([
                {
                    "type": "text", 
                    "text": "REFERENCE IMAGE:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": reference_images[0]}
                }
            ])
        
        content.append({
            "type": "text",
            "text": f"""
Respond with JSON only:
{{
  "category_match": true/false,
  "actual_content_type": "brief description",
  "confidence_assessment": "high/medium/low"
}}"""
        })
        
        return [HumanMessage(content=content)]
    
    def _parse_vision_response(self, response_text: str) -> Dict:
        """Parse vision response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        return {
            'category_match': 'true' in response_text.lower(),
            'actual_content_type': 'diagram' if 'diagram' in response_text.lower() else 'text',
            'confidence_assessment': 'medium'
        }
    
    def _generate_analysis(self, extraction_data: Dict, vision_results: List[Dict], 
                          current_params: Dict, debug: bool) -> Dict:
        """Generate simplified analysis"""
        total_analyzed = len(vision_results)
        correct_classifications = sum(1 for r in vision_results if r.get('category_match', False))
        accuracy_rate = correct_classifications / total_analyzed if total_analyzed > 0 else 0
        
        print(f"  📈 Accuracy: {accuracy_rate:.1%} ({correct_classifications}/{total_analyzed})")
        
        # Simple parameter recommendations
        param_recommendations = self._simple_parameter_recommendations(
            extraction_data, accuracy_rate, current_params
        )
        
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'vision_analysis_summary': {
                'total_items_analyzed': total_analyzed,
                'correct_classifications': correct_classifications,
                'accuracy_rate': accuracy_rate
            },
            'parameter_recommendations': param_recommendations,
            'llm_available': True,
            'llm_actually_used': True,
            'analysis_method': 'vision_based_simplified'
        }
    
    def _simple_parameter_recommendations(self, extraction_data: Dict, accuracy_rate: float, current_params: Dict) -> Dict:
        """Simple parameter recommendations"""
        recommendations = {}
        reasoning = []
        
        total_extracted = extraction_data.get('total_sections', 0)
        
        # Simple logic to prevent infinite loops
        if accuracy_rate < 0.3:
            # Very low accuracy - be more selective
            recommendations['confidence_threshold'] = 0.7
            reasoning.append(f"Low accuracy ({accuracy_rate:.1%}) - increasing selectivity")
        
        elif total_extracted > 20:
            # Too many extractions - increase thresholds
            recommendations['confidence_threshold'] = 0.6
            recommendations['min_region_size'] = 15000
            reasoning.append(f"High extraction count ({total_extracted}) - increasing filters")
        
        elif total_extracted < 3:
            # Too few extractions - decrease thresholds slightly
            recommendations['confidence_threshold'] = 0.4
            reasoning.append(f"Low extraction count ({total_extracted}) - reducing barriers")
        
        return {
            'adjustments': recommendations,
            'reasoning': '; '.join(reasoning) if reasoning else 'No major issues detected'
        }
    
    def _enhanced_fallback_analysis(self, extraction_data: Dict, debug: bool) -> Dict:
        """Enhanced fallback without vision"""
        print("  🔧 Running fallback analysis...")
        
        sections = extraction_data.get('extracted_sections', [])
        current_params = extraction_data.get('parameters_used', {})
        
        # Simple heuristic analysis
        param_recommendations = self._fallback_parameter_recommendations(sections, current_params)
        
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'fallback_analysis_summary': {
                'total_items_extracted': len(sections)
            },
            'parameter_recommendations': param_recommendations,
            'llm_available': False,
            'llm_actually_used': False,
            'analysis_method': 'enhanced_fallback'
        }
    
    def _fallback_parameter_recommendations(self, sections: List[Dict], current_params: Dict) -> Dict:
        """Fallback parameter recommendations"""
        total = len(sections)
        recommendations = {}
        
        if total < 3:
            recommendations['confidence_threshold'] = 0.4
            reasoning = f"Low extraction count ({total}) - reducing barriers"
        elif total > 25:
            recommendations['confidence_threshold'] = 0.7
            recommendations['min_region_size'] = 15000
            reasoning = f"High extraction count ({total}) - increasing selectivity"
        else:
            reasoning = "Moderate extraction levels - no major adjustments"
        
        return {
            'adjustments': recommendations,
            'reasoning': reasoning
        }
    
    def _timeout_fallback_analysis(self, extraction_data: Dict) -> Dict:
        """Analysis when timeout occurs"""
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'timeout_analysis': True,
            'parameter_recommendations': {
                'adjustments': {'confidence_threshold': 0.6},
                'reasoning': 'Analysis timed out - using safe defaults'
            },
            'llm_available': True,
            'llm_actually_used': False,
            'analysis_method': 'timeout_fallback'
        }
    
    def _error_fallback_analysis(self, extraction_data: Dict) -> Dict:
        """Analysis when error occurs"""
        return {
            'document_id': extraction_data.get('document_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'error_analysis': True,
            'parameter_recommendations': {
                'adjustments': {},
                'reasoning': 'Analysis failed - no changes recommended'
            },
            'llm_available': False,
            'llm_actually_used': False,
            'analysis_method': 'error_fallback'
        }
    
    def apply_recommendations(self, analysis_result: Dict, debug: bool = False) -> bool:
        """Apply parameter recommendations safely"""
        recommendations = analysis_result.get('parameter_recommendations', {})
        adjustments = recommendations.get('adjustments', {})
        
        if not adjustments:
            print("  ℹ️ No parameter adjustments recommended")
            return True
        
        print(f"  ⚙️ Applying {len(adjustments)} parameter adjustments...")
        
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
        
        # Apply adjustments with bounds
        changes_made = []
        for param_name, new_value in adjustments.items():
            try:
                # Validate bounds
                if param_name == 'confidence_threshold':
                    new_value = max(0.3, min(0.9, float(new_value)))
                elif param_name == 'min_region_size':
                    new_value = max(5000, min(25000, int(new_value)))
                elif param_name == 'similarity_threshold':
                    new_value = max(0.4, min(0.95, float(new_value)))
                
                old_value = current_params.get(param_name, 'N/A')
                if old_value != new_value:
                    current_params[param_name] = new_value
                    changes_made.append(f"{param_name}: {old_value} → {new_value}")
                    
                    if debug:
                        print(f"      ✓ {param_name}: {old_value} → {new_value}")
                        
            except Exception as e:
                print(f"      ✗ Invalid value for {param_name}: {e}")
        
        # Save parameters
        if changes_made:
            try:
                with open(param_file, 'w') as f:
                    json.dump(current_params, f, indent=2)
                print(f"    ✅ Applied {len(changes_made)} parameter changes")
                return True
            except Exception as e:
                print(f"    ❌ Error saving parameters: {e}")
                return False
        else:
            print("    ℹ️ No valid changes to apply")
            return True
    
    def save_analysis_log(self, analysis_result: Dict):
        """Save analysis log"""
        log_file = "feedback_log.json"
        
        try:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(analysis_result)
            
            # Keep only last 20 entries
            if len(logs) > 20:
                logs = logs[-20:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            print(f"  💾 Analysis saved to {log_file}")
            
        except Exception as e:
            print(f"  ⚠ Error saving analysis log: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM Feedback with Vision - FIXED VERSION")
    parser.add_argument("--analyze-and-apply", help="Analyze document ID and apply recommendations")
    parser.add_argument("--test-connection", action="store_true", help="Test Azure OpenAI connection")
    parser.add_argument("--show-log", action="store_true", help="Show recent feedback log")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.test_connection:
        print("🧪 Testing Azure OpenAI connection...")
        analyzer = VisionFeedbackAnalyzer()
        
        if analyzer.azure_available and analyzer.llm:
            try:
                messages = [HumanMessage(content="Test connection")]
                response = analyzer.llm.invoke(messages)
                print("✅ Azure OpenAI connection successful")
                return 0
            except Exception as e:
                print(f"❌ Connection test failed: {e}")
                return 1
        else:
            print("❌ Azure OpenAI not properly configured")
            return 1
    
    if args.show_log:
        try:
            with open("feedback_log.json") as f:
                logs = json.load(f)
            
            print(f"\n📊 Recent Feedback Log ({len(logs)} entries)")
            print("="*60)
            
            for log in logs[-5:]:  # Show last 5
                timestamp = log.get('timestamp', 'Unknown')
                doc_id = log.get('document_id', 'Unknown')
                method = log.get('analysis_method', 'Unknown')
                
                print(f"🕒 {timestamp}")
                print(f"📄 Document: {doc_id}")
                print(f"🔍 Method: {method}")
                
                if 'vision_analysis_summary' in log:
                    summary = log['vision_analysis_summary']
                    accuracy = summary.get('accuracy_rate', 0)
                    print(f"🎯 Accuracy: {accuracy:.1%}")
                
                print("-" * 40)
            
            return 0
            
        except FileNotFoundError:
            print("📄 No feedback log found")
            return 0
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return 1
    
    if args.analyze_and_apply:
        document_id = args.analyze_and_apply
        print(f"🔍 Analyzing document {document_id} with vision feedback...")
        
        try:
            analyzer = VisionFeedbackAnalyzer()
            
            # Perform analysis
            analysis_result = analyzer.analyze_extraction_with_vision(document_id, args.debug)
            
            # Apply recommendations
            success = analyzer.apply_recommendations(analysis_result, args.debug)
            
            # Save analysis log
            analyzer.save_analysis_log(analysis_result)
            
            if success:
                print("✅ Vision-based feedback analysis completed")
                return 0
            else:
                print("⚠️ Analysis completed with some issues")
                return 1
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    print("❓ No action specified. Use --help for options.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
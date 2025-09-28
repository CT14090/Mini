#!/usr/bin/env python3
# llm_feedback.py
"""
Enhanced LLM Feedback System with Azure OpenAI Vision Analysis
Provides intelligent feedback on extraction quality and parameter optimization
"""

import json
import os
import pathlib
import time
import argparse
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Azure OpenAI imports
AZURE_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import HumanMessage
    AZURE_AVAILABLE = True
except ImportError:
    print("⚠ Azure OpenAI not available - install with: pip install langchain-openai")

class EnhancedLLMFeedback:
    """Enhanced LLM feedback system with construction-focused vision analysis"""
    
    def __init__(self):
        self.feedback_log_file = "feedback_log.json"
        self.learning_params_file = "learning_parameters.json"
        self.azure_client = None
        
        # Initialize Azure OpenAI if available
        if AZURE_AVAILABLE:
            self._init_azure_client()
        
        # Load current learning parameters
        self.current_params = self._load_learning_parameters()
        
        print(f"✓ Enhanced feedback system initialized")
        if self.azure_client:
            print(f"  ✓ Azure OpenAI vision analysis available")
        else:
            print(f"  ⚠ Azure OpenAI not configured - using fallback analysis")
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
            
            if not all([endpoint, api_key, deployment]):
                print("  ⚠ Azure OpenAI credentials incomplete")
                return
            
            self.azure_client = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                azure_deployment=deployment,
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=2000
            )
            
        except Exception as e:
            print(f"  ⚠ Azure OpenAI initialization failed: {e}")
            self.azure_client = None
    
    def _load_learning_parameters(self) -> Dict:
        """Load current learning parameters"""
        default_params = {
            'confidence_threshold': 0.35,
            'min_region_size': 15000,
            'max_region_size': 2500000,
            'similarity_threshold': 0.35,
            'construction_bias': 2.0,
            'edge_density_range': [0.01, 0.25],
            'line_density_threshold': 0.005,
            'circular_elements_positive': True,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(self.learning_params_file):
                with open(self.learning_params_file) as f:
                    params = json.load(f)
                    # Ensure all default keys exist
                    for key, value in default_params.items():
                        if key not in params:
                            params[key] = value
                    return params
        except Exception as e:
            print(f"  ⚠ Error loading parameters: {e}")
        
        return default_params
    
    def analyze_extraction_quality(self, document_id: str, debug: bool = False) -> Dict:
        """Analyze extraction quality with enhanced vision feedback"""
        extraction_file = f"feedback_data/extraction_{document_id}.json"
        
        if not os.path.exists(extraction_file):
            raise FileNotFoundError(f"Extraction file not found: {extraction_file}")
        
        with open(extraction_file) as f:
            extraction_data = json.load(f)
        
        print(f"🤖 Analyzing extraction quality for {document_id}")
        
        # Enhanced analysis
        analysis_results = {
            'document_id': document_id,
            'timestamp': datetime.now().isoformat(),
            'analysis_method': 'enhanced_vision_feedback',
            'extraction_summary': self._analyze_extraction_summary(extraction_data),
            'vision_analysis_summary': {},
            'parameter_recommendations': {},
            'quality_metrics': {}
        }
        
        # Vision analysis of extracted items
        if self.azure_client:
            vision_results = self._run_vision_analysis(extraction_data, debug)
            analysis_results['vision_analysis_summary'] = vision_results
        else:
            # Fallback analysis without vision
            fallback_results = self._run_fallback_analysis(extraction_data, debug)
            analysis_results['vision_analysis_summary'] = fallback_results
        
        # Generate parameter recommendations
        recommendations = self._generate_parameter_recommendations(
            extraction_data, analysis_results['vision_analysis_summary'], debug
        )
        analysis_results['parameter_recommendations'] = recommendations
        
        # Calculate quality metrics
        analysis_results['quality_metrics'] = self._calculate_quality_metrics(
            extraction_data, analysis_results['vision_analysis_summary']
        )
        
        # Log analysis results
        self._log_feedback_analysis(analysis_results)
        
        if debug:
            print(f"  Analysis complete - quality metrics calculated")
        
        return analysis_results
    
    def _analyze_extraction_summary(self, extraction_data: Dict) -> Dict:
        """Analyze basic extraction statistics"""
        summary = extraction_data.get('extraction_summary', {})
        sections = extraction_data.get('extracted_sections', [])
        
        return {
            'total_items': len(sections),
            'pages_processed': summary.get('pages_processed', 0),
            'visual_regions_analyzed': summary.get('visual_regions_analyzed', 0),
            'avg_confidence': summary.get('avg_confidence', 0),
            'high_confidence_items': summary.get('high_confidence_items', 0),
            'extraction_rate': summary.get('extraction_rate_per_page', 0),
            'detection_methods': summary.get('detection_method_breakdown', {}),
            'content_types': summary.get('content_type_breakdown', {})
        }
    
    def _run_vision_analysis(self, extraction_data: Dict, debug: bool = False) -> Dict:
        """Run Azure OpenAI vision analysis on extracted items"""
        sections = extraction_data.get('extracted_sections', [])
        characteristic = extraction_data.get('target_characteristic', 'unknown')
        
        if not sections:
            return {
                'total_items_analyzed': 0,
                'vision_available': True,
                'analysis_status': 'no_items_to_analyze'
            }
        
        print(f"  🔍 Running Azure vision analysis on {len(sections)} items...")
        
        # Analyze up to 10 items to avoid token limits
        items_to_analyze = sections[:10]
        vision_results = []
        
        for i, section in enumerate(items_to_analyze):
            if debug:
                print(f"    Analyzing item {i+1}/{len(items_to_analyze)}...")
            
            try:
                # Get image data
                data_uri = section.get('data_uri', '')
                if not data_uri or not data_uri.startswith('data:image'):
                    continue
                
                # Create vision analysis prompt
                prompt = self._create_vision_analysis_prompt(section, characteristic)
                
                # Prepare message with image
                message_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
                
                message = HumanMessage(content=message_content)
                
                # Get analysis from Azure OpenAI
                response = self.azure_client.invoke([message])
                analysis_text = response.content
                
                # Parse analysis results
                analysis = self._parse_vision_analysis(analysis_text, section)
                analysis['item_index'] = i
                vision_results.append(analysis)
                
                if debug:
                    print(f"      ✓ Item {i+1}: {analysis.get('quality_assessment', 'analyzed')}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                if debug:
                    print(f"      ❌ Error analyzing item {i+1}: {e}")
                vision_results.append({
                    'item_index': i,
                    'error': str(e),
                    'quality_assessment': 'error'
                })
        
        # Compile vision analysis summary
        return self._compile_vision_summary(vision_results, len(sections))
    
    def _create_vision_analysis_prompt(self, section: Dict, characteristic: str) -> str:
        """Create detailed vision analysis prompt for Azure-first extractions"""
        content_type = section.get('region_metadata', {}).get('content_type', 'unknown')
        confidence = section.get('confidence', 0)
        detection_method = section.get('region_metadata', {}).get('detection_method', 'unknown')
        azure_description = section.get('region_metadata', {}).get('azure_description', '')
        
        prompt = f"""
Analyze this extracted construction document image for {characteristic.replace('_', ' ')} content.

CONTEXT:
- Target characteristic: {characteristic.replace('_', ' ').title()}
- Detection method: {detection_method}
- System confidence: {confidence:.3f}
- Azure description: {azure_description}
- Source: Construction technical document

ANALYSIS CRITERIA:

1. CONTENT RELEVANCE (Critical):
   - Does this image contain {characteristic.replace('_', ' ')} information?
   - Is it a technical diagram, table, or specification relevant to {characteristic}?
   - How well does it match the target characteristic?
   - Rate relevance: HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, NOT_RELEVANT

2. VISUAL QUALITY (Important):
   - Is the image clear and readable?
   - Can technical details be discerned?
   - Is the extraction well-bounded (not cut off)?
   - Rate quality: EXCELLENT, GOOD, FAIR, POOR

3. CONSTRUCTION AUTHENTICITY (Essential):
   - Is this legitimate construction/engineering content?
   - Are there technical drawings, specifications, or data relevant to construction?
   - Is this clearly NOT a logo, badge, header, or decorative element?
   - Rate authenticity: CLEARLY_CONSTRUCTION, LIKELY_CONSTRUCTION, UNCERTAIN, NOT_CONSTRUCTION

4. EXTRACTION ACCURACY (Critical):
   - Was the right visual content extracted?
   - Is the extracted region appropriately sized and positioned?
   - Does it contain the complete relevant information?
   - Rate accuracy: EXCELLENT, GOOD, FAIR, POOR

5. TRAINING DATA ALIGNMENT:
   - How well does this match typical {characteristic.replace('_', ' ')} content?
   - Would this be a good training example for {characteristic}?
   - Rate alignment: PERFECT_MATCH, GOOD_MATCH, PARTIAL_MATCH, NO_MATCH

Provide your analysis in this format:
RELEVANCE: [rating]
QUALITY: [rating]  
CONSTRUCTION: [rating]
ACCURACY: [rating]
TRAINING_ALIGNMENT: [rating]
CONFIDENCE: [0.0-1.0]
NOTES: [Brief explanation focusing on why this does/doesn't match {characteristic}]
"""
        return prompt
    
    def _parse_vision_analysis(self, analysis_text: str, section: Dict) -> Dict:
        """Parse vision analysis response into structured data"""
        analysis = {
            'relevance': 'UNCERTAIN',
            'quality': 'FAIR',
            'construction': 'UNCERTAIN', 
            'accuracy': 'FAIR',
            'confidence': 0.5,
            'notes': analysis_text[:200],  # First 200 chars as notes
            'full_analysis': analysis_text
        }
        
        try:
            lines = analysis_text.upper().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('RELEVANCE:'):
                    analysis['relevance'] = line.split(':', 1)[1].strip()
                elif line.startswith('QUALITY:'):
                    analysis['quality'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONSTRUCTION:'):
                    analysis['construction'] = line.split(':', 1)[1].strip()
                elif line.startswith('ACCURACY:'):
                    analysis['accuracy'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    conf_str = line.split(':', 1)[1].strip()
                    try:
                        analysis['confidence'] = float(conf_str)
                    except:
                        analysis['confidence'] = 0.5
                elif line.startswith('NOTES:'):
                    analysis['notes'] = line.split(':', 1)[1].strip()[:300]
        
        except Exception as e:
            analysis['parsing_error'] = str(e)
        
        return analysis
    
    def _compile_vision_summary(self, vision_results: List[Dict], total_items: int) -> Dict:
        """Compile vision analysis results into summary"""
        if not vision_results:
            return {
                'total_items_analyzed': 0,
                'vision_available': True,
                'analysis_status': 'no_successful_analysis'
            }
        
        # Count quality assessments
        relevance_counts = {}
        quality_counts = {}
        construction_counts = {}
        accuracy_counts = {}
        confidences = []
        
        successful_analyses = [r for r in vision_results if 'error' not in r]
        
        for result in successful_analyses:
            relevance = result.get('relevance', 'UNCERTAIN')
            quality = result.get('quality', 'FAIR')
            construction = result.get('construction', 'UNCERTAIN')
            accuracy = result.get('accuracy', 'FAIR')
            confidence = result.get('confidence', 0.5)
            
            relevance_counts[relevance] = relevance_counts.get(relevance, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            construction_counts[construction] = construction_counts.get(construction, 0) + 1
            accuracy_counts[accuracy] = accuracy_counts.get(accuracy, 0) + 1
            confidences.append(confidence)
        
        # Calculate metrics
        total_analyzed = len(successful_analyses)
        
        # Relevance metrics
        highly_relevant = relevance_counts.get('HIGHLY_RELEVANT', 0)
        somewhat_relevant = relevance_counts.get('SOMEWHAT_RELEVANT', 0)
        not_relevant = relevance_counts.get('NOT_RELEVANT', 0)
        
        relevance_rate = (highly_relevant + somewhat_relevant) / max(1, total_analyzed)
        
        # Construction authenticity metrics  
        clearly_construction = construction_counts.get('CLEARLY_CONSTRUCTION', 0)
        likely_construction = construction_counts.get('LIKELY_CONSTRUCTION', 0)
        not_construction = construction_counts.get('NOT_CONSTRUCTION', 0)
        
        construction_rate = (clearly_construction + likely_construction) / max(1, total_analyzed)
        
        # Quality metrics
        excellent_quality = quality_counts.get('EXCELLENT', 0)
        good_quality = quality_counts.get('GOOD', 0)
        poor_quality = quality_counts.get('POOR', 0)
        
        quality_rate = (excellent_quality + good_quality) / max(1, total_analyzed)
        
        # Accuracy metrics
        excellent_accuracy = accuracy_counts.get('EXCELLENT', 0)
        good_accuracy = accuracy_counts.get('GOOD', 0)
        poor_accuracy = accuracy_counts.get('POOR', 0)
        
        accuracy_rate = (excellent_accuracy + good_accuracy) / max(1, total_analyzed)
        
        return {
            'total_items_analyzed': total_analyzed,
            'total_items_in_extraction': total_items,
            'vision_available': True,
            'analysis_status': 'completed',
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.5,
            'relevance_rate': relevance_rate,
            'construction_authenticity_rate': construction_rate,
            'visual_quality_rate': quality_rate,
            'extraction_accuracy_rate': accuracy_rate,
            'detailed_counts': {
                'relevance': relevance_counts,
                'quality': quality_counts,
                'construction': construction_counts,
                'accuracy': accuracy_counts
            },
            'errors': len(vision_results) - total_analyzed
        }
    
    def _run_fallback_analysis(self, extraction_data: Dict, debug: bool = False) -> Dict:
        """Fallback analysis when Azure OpenAI is not available"""
        sections = extraction_data.get('extracted_sections', [])
        
        if debug:
            print(f"  📊 Running fallback analysis on {len(sections)} items...")
        
        # Analyze based on metadata and confidence scores
        high_confidence = sum(1 for s in sections if s.get('confidence', 0) > 0.6)
        medium_confidence = sum(1 for s in sections if 0.4 <= s.get('confidence', 0) <= 0.6)
        low_confidence = sum(1 for s in sections if s.get('confidence', 0) < 0.4)
        
        # Analyze detection methods
        detection_methods = {}
        for section in sections:
            method = section.get('region_metadata', {}).get('detection_method', 'unknown')
            detection_methods[method] = detection_methods.get(method, 0) + 1
        
        # Estimate quality based on confidence and detection methods
        total_items = len(sections)
        estimated_accuracy = (high_confidence * 1.0 + medium_confidence * 0.7 + low_confidence * 0.4) / max(1, total_items)
        
        return {
            'total_items_analyzed': total_items,
            'vision_available': False,
            'analysis_status': 'fallback_analysis',
            'estimated_accuracy_rate': estimated_accuracy,
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'detection_method_analysis': detection_methods,
            'notes': 'Analysis based on confidence scores and metadata (Azure OpenAI not available)'
        }
    
    def _generate_parameter_recommendations(self, extraction_data: Dict, vision_analysis: Dict, debug: bool = False) -> Dict:
        """Generate intelligent parameter recommendations for Azure-first system"""
        if debug:
            print(f"  Generating Azure-optimized parameter recommendations...")
        
        recommendations = {
            'adjustments': {},
            'reasoning': '',
            'priority': 'medium',
            'confidence': 0.7
        }
        
        # Get extraction performance
        total_items = len(extraction_data.get('extracted_sections', []))
        azure_calls = extraction_data.get('extraction_summary', {}).get('azure_api_calls', 0)
        
        reasoning_parts = []
        
        # No extractions case - common with restrictive Azure prompts
        if total_items == 0:
            if azure_calls > 0:
                # Azure was used but found nothing - prompts might be too restrictive
                recommendations['adjustments']['azure_prompt_mode'] = 'generous'
                recommendations['adjustments']['confidence_threshold'] = 0.3
                recommendations['adjustments']['region_selection_mode'] = 'inclusive'
                reasoning_parts.append("No extractions with Azure - switching to generous mode")
                recommendations['priority'] = 'critical'
            else:
                # Fallback mode
                recommendations['adjustments']['min_region_size'] = 8000
                recommendations['adjustments']['confidence_threshold'] = 0.25
                reasoning_parts.append("No extractions in fallback mode - lowering thresholds")
                recommendations['priority'] = 'high'
        
        # Low extraction rate (less than 1 per page)
        elif azure_calls > 0:
            pages_processed = extraction_data.get('pages_processed', 1)
            extraction_rate = total_items / pages_processed
            
            if extraction_rate < 1.0:
                recommendations['adjustments']['azure_prompt_mode'] = 'inclusive'
                recommendations['adjustments']['confidence_threshold'] = 0.4
                reasoning_parts.append(f"Low extraction rate ({extraction_rate:.1f}/page) - being more inclusive")
                recommendations['priority'] = 'high'
            elif extraction_rate > 5.0:
                recommendations['adjustments']['azure_prompt_mode'] = 'selective'
                recommendations['adjustments']['confidence_threshold'] = 0.7
                reasoning_parts.append(f"High extraction rate ({extraction_rate:.1f}/page) - being more selective")
        
        # Vision analysis feedback (if available)
        if vision_analysis.get('vision_available', False):
            relevance_rate = vision_analysis.get('relevance_rate', 0)
            accuracy_rate = vision_analysis.get('extraction_accuracy_rate', 0)
            
            if relevance_rate < 0.4:
                recommendations['adjustments']['training_data_integration'] = 'enhanced'
                recommendations['adjustments']['azure_prompt_mode'] = 'training_focused'
                reasoning_parts.append(f"Low relevance ({relevance_rate:.1%}) - focusing on training data alignment")
                recommendations['priority'] = 'critical'
            
            if accuracy_rate < 0.5:
                recommendations['adjustments']['coordinate_parsing'] = 'flexible'
                recommendations['adjustments']['region_validation'] = 'lenient'
                reasoning_parts.append(f"Low accuracy ({accuracy_rate:.1%}) - improving region handling")
        
        # Training data alignment
        characteristic = extraction_data.get('target_characteristic', '')
        if characteristic and total_items == 0:
            recommendations['adjustments']['characteristic_terms'] = 'expanded'
            recommendations['adjustments']['visual_pattern_matching'] = 'flexible'
            reasoning_parts.append(f"Zero extractions for {characteristic} - expanding search terms")
        
        # Set reasoning
        recommendations['reasoning'] = '; '.join(reasoning_parts) if reasoning_parts else 'System performing well - no major adjustments needed'
        
        # Set confidence based on available data
        if azure_calls > 0 and vision_analysis.get('vision_available'):
            recommendations['confidence'] = 0.9
        elif azure_calls > 0:
            recommendations['confidence'] = 0.8
        else:
            recommendations['confidence'] = 0.6
        
        if debug and recommendations['adjustments']:
            print(f"    Recommended {len(recommendations['adjustments'])} Azure-specific adjustments")
        
        return recommendations = vision_analysis.get('estimated_accuracy_rate', 0.5)
            
            if estimated_accuracy < 0.4:
                recommendations['adjustments']['confidence_threshold'] = max(0.25, self.current_params['confidence_threshold'] - 0.1)
                recommendations['reasoning'] = f"Lowered confidence threshold due to low estimated accuracy ({estimated_accuracy:.1%})"
                recommendations['priority'] = 'high'
            elif estimated_accuracy > 0.8:
                recommendations['adjustments']['confidence_threshold'] = min(0.6, self.current_params['confidence_threshold'] + 0.05)
                recommendations['reasoning'] = f"Raised confidence threshold due to high estimated accuracy ({estimated_accuracy:.1%})"
            else:
                recommendations['reasoning'] = "No adjustments recommended based on fallback analysis"
        
        # No extractions case
        if total_items == 0:
            recommendations['adjustments']['confidence_threshold'] = max(0.2, self.current_params['confidence_threshold'] - 0.15)
            recommendations['adjustments']['similarity_threshold'] = max(0.2, self.current_params['similarity_threshold'] - 0.15)
            recommendations['adjustments']['construction_bias'] = min(3.0, self.current_params['construction_bias'] + 0.5)
            recommendations['reasoning'] = "Significantly lowered thresholds due to no extractions"
            recommendations['priority'] = 'critical'
            recommendations['confidence'] = 0.8
        
        if debug and recommendations['adjustments']:
            print(f"    Recommended adjustments: {len(recommendations['adjustments'])} parameters")
        
        return recommendations
    
    def _calculate_quality_metrics(self, extraction_data: Dict, vision_analysis: Dict) -> Dict:
        """Calculate comprehensive quality metrics"""
        sections = extraction_data.get('extracted_sections', [])
        summary = extraction_data.get('extraction_summary', {})
        
        metrics = {
            'extraction_efficiency': 0,
            'confidence_quality': 0,
            'detection_diversity': 0,
            'overall_quality_score': 0
        }
        
        if not sections:
            return metrics
        
        # Extraction efficiency (items per page)
        pages_processed = summary.get('pages_processed', 1)
        extraction_rate = len(sections) / pages_processed
        metrics['extraction_efficiency'] = min(1.0, extraction_rate / 2)  # Normalize to expected 2 items per page
        
        # Confidence quality
        confidences = [s.get('confidence', 0) for s in sections]
        avg_confidence = sum(confidences) / len(confidences)
        metrics['confidence_quality'] = avg_confidence
        
        # Detection method diversity
        detection_methods = set()
        for section in sections:
            method = section.get('region_metadata', {}).get('detection_method', 'unknown')
            detection_methods.add(method)
        
        metrics['detection_diversity'] = min(1.0, len(detection_methods) / 3)  # Normalize to 3 methods
        
        # Overall quality score
        if vision_analysis.get('vision_available', False):
            # Use vision analysis for overall score
            accuracy_rate = vision_analysis.get('extraction_accuracy_rate', 0.5)
            construction_rate = vision_analysis.get('construction_authenticity_rate', 0.5)
            relevance_rate = vision_analysis.get('relevance_rate', 0.5)
            
            metrics['overall_quality_score'] = (
                accuracy_rate * 0.4 +           # 40% weight on accuracy
                construction_rate * 0.3 +       # 30% weight on construction authenticity  
                relevance_rate * 0.2 +          # 20% weight on relevance
                metrics['confidence_quality'] * 0.1  # 10% weight on confidence
            )
        else:
            # Fallback scoring
            estimated_accuracy = vision_analysis.get('estimated_accuracy_rate', 0.5)
            metrics['overall_quality_score'] = (
                estimated_accuracy * 0.5 +
                metrics['confidence_quality'] * 0.3 +
                metrics['extraction_efficiency'] * 0.2
            )
        
        return metrics
    
    def apply_parameter_recommendations(self, document_id: str, debug: bool = False) -> bool:
        """Apply parameter recommendations from analysis"""
        try:
            # Load analysis results
            feedback_log = self._load_feedback_log()
            
            # Find the latest analysis for this document
            latest_analysis = None
            for entry in reversed(feedback_log):
                if entry.get('document_id') == document_id:
                    latest_analysis = entry
                    break
            
            if not latest_analysis:
                print(f"  ❌ No analysis found for {document_id}")
                return False
            
            recommendations = latest_analysis.get('parameter_recommendations', {})
            adjustments = recommendations.get('adjustments', {})
            
            if not adjustments:
                print(f"  ℹ️  No parameter adjustments recommended")
                return True
            
            print(f"  🔧 Applying {len(adjustments)} parameter adjustments...")
            
            # Apply adjustments to current parameters
            for param, new_value in adjustments.items():
                old_value = self.current_params.get(param, 'unknown')
                self.current_params[param] = new_value
                
                if debug:
                    print(f"    {param}: {old_value} → {new_value}")
            
            # Update timestamp
            self.current_params['last_updated'] = datetime.now().isoformat()
            self.current_params['last_analysis_document'] = document_id
            
            # Save updated parameters
            self._save_learning_parameters()
            
            print(f"  ✅ Parameter adjustments applied")
            print(f"      Reasoning: {recommendations.get('reasoning', 'No reasoning provided')}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error applying recommendations: {e}")
            return False
    
    def _load_feedback_log(self) -> List[Dict]:
        """Load feedback log entries"""
        try:
            if os.path.exists(self.feedback_log_file):
                with open(self.feedback_log_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return []
    
    def _log_feedback_analysis(self, analysis: Dict):
        """Log feedback analysis to file"""
        try:
            # Load existing log
            log_entries = self._load_feedback_log()
            
            # Add new entry
            log_entries.append(analysis)
            
            # Keep only last 50 entries
            log_entries = log_entries[-50:]
            
            # Save log
            with open(self.feedback_log_file, 'w') as f:
                json.dump(log_entries, f, indent=2)
                
        except Exception as e:
            print(f"  ⚠ Error logging analysis: {e}")
    
    def _save_learning_parameters(self):
        """Save learning parameters to file"""
        try:
            with open(self.learning_params_file, 'w') as f:
                json.dump(self.current_params, f, indent=2)
        except Exception as e:
            print(f"  ⚠ Error saving parameters: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced LLM Feedback System")
    parser.add_argument("--analyze", help="Analyze extraction quality for document ID")
    parser.add_argument("--apply", help="Apply recommendations for document ID") 
    parser.add_argument("--analyze-and-apply", help="Analyze and apply recommendations for document ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show-params", action="store_true", help="Show current learning parameters")
    parser.add_argument("--reset-params", action="store_true", help="Reset parameters to defaults")
    
    args = parser.parse_args()
    
    if args.show_params:
        feedback = EnhancedLLMFeedback()
        print("\n📊 Current Learning Parameters:")
        for key, value in feedback.current_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        return 0
    
    if args.reset_params:
        import os
        params_file = "learning_parameters.json"
        if os.path.exists(params_file):
            os.remove(params_file)
            print("✅ Learning parameters reset to defaults")
        else:
            print("ℹ️  No parameters file to reset")
        return 0
    
    if not any([args.analyze, args.apply, args.analyze_and_apply]):
        parser.error("One of --analyze, --apply, or --analyze-and-apply is required")
    
    try:
        feedback = EnhancedLLMFeedback()
        
        if args.analyze or args.analyze_and_apply:
            doc_id = args.analyze or args.analyze_and_apply
            print(f"🤖 Enhanced Vision Feedback Analysis")
            print(f"Document ID: {doc_id}")
            
            analysis = feedback.analyze_extraction_quality(doc_id, debug=args.debug)
            
            # Print summary
            vision_summary = analysis.get('vision_analysis_summary', {})
            quality_metrics = analysis.get('quality_metrics', {})
            
            if vision_summary.get('vision_available', False):
                print(f"\n📊 Vision Analysis Results:")
                print(f"  Items analyzed: {vision_summary.get('total_items_analyzed', 0)}")
                print(f"  Accuracy: {vision_summary.get('extraction_accuracy_rate', 0):.1%}")
                print(f"  Construction authenticity: {vision_summary.get('construction_authenticity_rate', 0):.1%}")
                print(f"  Relevance: {vision_summary.get('relevance_rate', 0):.1%}")
                print(f"  Visual quality: {vision_summary.get('visual_quality_rate', 0):.1%}")
            else:
                print(f"\n📊 Fallback Analysis Results:")
                print(f"  Items analyzed: {vision_summary.get('total_items_analyzed', 0)}")
                print(f"  Estimated accuracy: {vision_summary.get('estimated_accuracy_rate', 0):.1%}")
            
            print(f"\n🎯 Quality Metrics:")
            print(f"  Overall quality score: {quality_metrics.get('overall_quality_score', 0):.1%}")
            print(f"  Confidence quality: {quality_metrics.get('confidence_quality', 0):.3f}")
            print(f"  Extraction efficiency: {quality_metrics.get('extraction_efficiency', 0):.3f}")
            
            # Show recommendations
            recommendations = analysis.get('parameter_recommendations', {})
            adjustments = recommendations.get('adjustments', {})
            
            if adjustments:
                print(f"\n🔧 Parameter Recommendations ({recommendations.get('priority', 'medium')} priority):")
                for param, value in adjustments.items():
                    print(f"  {param}: {value}")
                print(f"  Reasoning: {recommendations.get('reasoning', 'No reasoning provided')}")
            else:
                print(f"\nℹ️  No parameter adjustments recommended")
        
        if args.apply or args.analyze_and_apply:
            doc_id = args.apply or args.analyze_and_apply
            print(f"\n🔧 Applying Parameter Recommendations")
            
            success = feedback.apply_parameter_recommendations(doc_id, debug=args.debug)
            
            if success:
                print(f"✅ Recommendations applied successfully")
            else:
                print(f"❌ Failed to apply recommendations")
                return 1
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"Make sure to run extraction first:")
        print(f"  python adaptive_agent.py --source document.pdf --characteristic anchors")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
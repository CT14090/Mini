#!/usr/bin/env python3
"""
Enhanced LLM Feedback System with Reference Data and Original PDF Access
Provides context-aware feedback using reference images and original document
"""
import json
import os
import sys
import pathlib
import argparse
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✓ Environment variables loaded")
except Exception as e:
    print(f"⚠ Environment loading issue: {e}")

    
# LangChain imports
LANGCHAIN_OK = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_OK = True
    print("✓ LangChain Azure OpenAI available for enhanced feedback")
except ImportError as e:
    print(f"⚠ LangChain not available: {e}")

class EnhancedCharacteristicLLMAnalyzer:
    """Enhanced LLM analyzer with reference data and original PDF access"""
    
    def __init__(self, characteristic: str):
        self.characteristic = characteristic
        
        # Check if LangChain is available
        if not LANGCHAIN_OK:
            print("❌ LangChain is required for enhanced LLM feedback!")
            print("   Install with: pip install langchain-openai langchain-core")
            raise RuntimeError("LangChain required but not available")
        
        # Check Azure OpenAI configuration
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Azure OpenAI configuration missing: {missing_vars}")
            raise RuntimeError("Azure OpenAI configuration required but missing")
        
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=2000
            )
            print(f"✓ Enhanced Azure OpenAI initialized for {characteristic} analysis")
        except Exception as e:
            print(f"❌ Azure OpenAI initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {e}")
        
        # Load reference data
        self.reference_data = self._load_reference_data()
        
        # Enhanced analysis prompts
        self.analysis_prompts = self._get_enhanced_analysis_prompts()
    
    def _load_reference_data(self) -> Dict:
        """Load reference data for the characteristic"""
        reference_dir = pathlib.Path("labeled_data") / self.characteristic
        
        if not reference_dir.exists():
            print(f"⚠ No reference data directory found for {self.characteristic}")
            return {'descriptions': {}, 'images': [], 'context': 'No reference data available'}
        
        # Load descriptions
        descriptions = {}
        desc_file = reference_dir / "descriptions.json"
        if desc_file.exists():
            try:
                with open(desc_file) as f:
                    descriptions = json.load(f)
            except Exception as e:
                print(f"⚠ Error loading reference descriptions: {e}")
        
        # Get image list
        image_files = list(reference_dir.glob("*.{jpg,jpeg,png}"))
        
        reference_context = self._create_reference_context(descriptions, image_files)
        
        return {
            'descriptions': descriptions,
            'images': [str(f.name) for f in image_files],
            'context': reference_context,
            'total_references': len(image_files)
        }
    
    def _create_reference_context(self, descriptions: Dict, image_files: List) -> str:
        """Create context description of reference data"""
        context_parts = []
        
        if descriptions:
            context_parts.append("REFERENCE DESCRIPTIONS:")
            for ref_type, desc in descriptions.items():
                context_parts.append(f"- {ref_type}: {desc}")
        
        if image_files:
            context_parts.append(f"\nREFERENCE IMAGES AVAILABLE: {len(image_files)}")
            for img_file in image_files[:5]:  # Show first 5
                context_parts.append(f"- {img_file.stem}")
            if len(image_files) > 5:
                context_parts.append(f"- ... and {len(image_files) - 5} more")
        
        return '\n'.join(context_parts) if context_parts else "No reference data available"
    
    def _get_enhanced_analysis_prompts(self) -> Dict[str, str]:
        """Get enhanced analysis prompts with reference data context"""
        return {
            'anchors': f"""
You are analyzing window anchor extraction results with access to reference data and the original PDF.

REFERENCE DATA CONTEXT:
{self.reference_data.get('context', 'No reference data available')}

ANCHOR TYPES TO IDENTIFY (your primary focus):
- Directly Into Concrete: Anchors/screws going directly into concrete substrate
- Directly Into Wood: Screws/fasteners going directly into wood framing
- Into Wood via 1By Buck: Anchors going through 1x wood buck into substrate
- Into Concrete via 1By Buck: Anchors going through 1x wood buck into concrete  
- Into Concrete via 2By Buck: Anchors going through 2x wood buck into concrete
- Self Drilling Screws Into Metal Structures: Self-drilling screws for metal framing

ENHANCED EVALUATION CRITERIA:
1. Were the correct anchor types identified based on the text summary and context?
2. Do extracted images actually show anchor details, installation methods, or connection points?
3. Do extracted tables contain anchor specifications, schedules, or fastener data?
4. Was irrelevant content (logos, general diagrams) properly filtered out?
5. Does the extraction align with what you know about proper anchor documentation?

REFERENCE COMPARISON:
Compare extracted content with the reference descriptions above. Are the extracted items 
actually relevant to window anchoring systems, or were general construction details 
mistakenly classified as anchor-related?

PARAMETER RECOMMENDATIONS:
Focus on improving precision - it's better to miss some anchor details than to extract 
irrelevant content. Adjust thresholds to be more selective about what qualifies as 
anchor-related content.
""",
            
            'glazing': f"""
You are analyzing window glazing extraction results with access to reference data.

REFERENCE DATA CONTEXT:
{self.reference_data.get('context', 'No reference data available')}

GLAZING CHARACTERISTICS TO IDENTIFY:
- Glass type and thickness specifications
- Low-E coatings and performance ratings  
- Insulated glass unit (IGU/IGP) configurations
- Laminated vs tempered vs annealed glass
- Double/triple glazing specifications
- Thermal and optical properties

ENHANCED EVALUATION CRITERIA:
1. Were glass specifications correctly identified from text and tables?
2. Do extracted images show actual glazing details, sections, or assemblies?
3. Are glazing performance data and ratings properly captured?
4. Was irrelevant content properly filtered out?

Compare with reference data to ensure extracted content is truly glazing-related.
""",
            
            'impact_rating': f"""
You are analyzing window impact rating extraction results with access to reference data.

REFERENCE DATA CONTEXT:
{self.reference_data.get('context', 'No reference data available')}

IMPACT RATING TYPES TO IDENTIFY:
- Small Missile Impact ratings and test results
- Large Missile Impact ratings and test results
- Both Missile Impact comprehensive ratings
- ASTM/AAMA test compliance documentation
- Hurricane/storm ratings and certifications

ENHANCED EVALUATION CRITERIA:
1. Were impact ratings correctly identified and categorized?
2. Do extracted images show test results, certifications, or impact data?
3. Are compliance standards properly referenced?
4. Was general content mistaken for impact-specific information?

Compare with reference data to verify accuracy of impact-related classifications.
""",
            
            'design_pressure': f"""
You are analyzing window design pressure extraction results with access to reference data.

REFERENCE DATA CONTEXT:
{self.reference_data.get('context', 'No reference data available')}

DESIGN PRESSURE DATA TO IDENTIFY:
- Design pressure ratings (DP values) and specifications
- Positive and negative pressure ratings
- Wind load specifications (PSF, PA, KPA)
- Structural load ratings and performance data
- Pressure rating tables and performance charts

ENHANCED EVALUATION CRITERIA:
1. Were design pressure values and ratings accurately extracted?
2. Do extracted tables contain actual pressure data and specifications?
3. Are structural performance metrics properly identified?
4. Was general structural information mistaken for pressure-specific data?

Compare with reference data to ensure pressure-related content accuracy.
"""
        }
    
    def analyze_enhanced_extraction(self, document_id: str, source_pdf: str = None, debug: bool = False) -> Optional[Dict]:
        """Analyze extraction with enhanced context including original PDF access"""
        try:
            # Load extraction data
            extraction_file = f"feedback_data/{self.characteristic}_extraction_{document_id}.json"
            if not os.path.exists(extraction_file):
                print(f"❌ Enhanced extraction file not found: {extraction_file}")
                return None
            
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            # Prepare enhanced analysis data
            analysis_data = self._prepare_enhanced_analysis_data(extraction_data, source_pdf, debug)
            
            # Create enhanced analysis prompt
            system_prompt = self.analysis_prompts.get(self.characteristic, 
                                                    self.analysis_prompts['anchors'])
            
            user_prompt = f"""
ENHANCED {self.characteristic.upper()} EXTRACTION ANALYSIS

DOCUMENT PROCESSING SUMMARY:
- Original PDF: {os.path.basename(source_pdf) if source_pdf else 'Not provided'}
- Pages processed: {extraction_data.get('enhancement_features', {}).get('pages_skipped', 0)} pages skipped
- Text summarization: {'Enabled' if extraction_data.get('text_summary', {}).get('summary') else 'Disabled'}
- Reference matching: {'Enabled' if extraction_data.get('reference_data_used', {}).get('total_references', 0) > 0 else 'Disabled'}

EXTRACTION RESULTS:
{analysis_data}

CRITICAL ANALYSIS NEEDED:
1. RELEVANCE ACCURACY (1-5 scale):
   - Are extracted images actually showing {self.characteristic.replace('_', ' ')} details?
   - Are extracted tables truly {self.characteristic.replace('_', ' ')}-related?
   - Was irrelevant content properly filtered out?

2. EXTRACTION QUALITY (1-5 scale):
   - Completeness: Was important {self.characteristic.replace('_', ' ')} information captured?
   - Precision: Were false positives minimized?
   - Context: Does the extraction make sense given the document type?

3. REFERENCE ALIGNMENT:
   - How well do extracted items align with the reference descriptions provided?
   - Are there obvious misclassifications based on your knowledge of {self.characteristic.replace('_', ' ')}?

4. ENHANCED PARAMETER RECOMMENDATIONS:
   Current parameters: {json.dumps(extraction_data.get('parameters_used', {}), indent=2)}
   
   Suggest specific adjustments to improve:
   - content_classification_threshold (currently {extraction_data.get('parameters_used', {}).get('content_classification_threshold', 0.2)})
   - confidence_threshold (currently {extraction_data.get('parameters_used', {}).get('confidence_threshold', 0.3)})
   - skip_pages (currently {extraction_data.get('parameters_used', {}).get('skip_pages', 3)})
   - image_size_min (currently {extraction_data.get('parameters_used', {}).get('image_size_min', 100)})

Respond in JSON format:
{{
    "extraction_quality": {{
        "relevance_accuracy": <1-5>,
        "completeness": <1-5>, 
        "precision": <1-5>,
        "context_appropriateness": <1-5>
    }},
    "enhanced_feedback": {{
        "correctly_identified": ["item1", "item2"],
        "incorrectly_classified": ["item3", "item4"],
        "missing_content": ["item5", "item6"],
        "reference_alignment": "assessment compared to reference data"
    }},
    "parameter_recommendations": {{
        "content_classification_threshold": <new_value>,
        "confidence_threshold": <new_value>,
        "skip_pages": <new_value>,
        "image_size_min": <new_value>,
        "reasoning": "detailed explanation of recommended changes based on analysis"
    }}
}}
"""
            
            if debug:
                print(f"🤖 Sending enhanced {self.characteristic} analysis to LLM...")
                print(f"Analysis data length: {len(analysis_data)} characters")
                print(f"Reference context: {len(self.reference_data.get('context', ''))} characters")
            
            # Get enhanced LLM analysis
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if debug:
                print(f"✓ Enhanced LLM response received: {len(response.content)} characters")
            
            # Parse JSON response with enhanced error handling
            try:
                # Clean the response - remove markdown code blocks if present
                response_content = response.content.strip()
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                elif response_content.startswith('```'):
                    response_content = response_content.replace('```', '').strip()
                
                analysis_result = json.loads(response_content)
                return analysis_result
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse enhanced LLM response as JSON: {e}")
                if debug:
                    print(f"Raw response: {response.content[:500]}...")
                return None
                
        except Exception as e:
            print(f"❌ Enhanced LLM analysis error: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _prepare_enhanced_analysis_data(self, extraction_data: Dict, source_pdf: str, debug: bool = False) -> str:
        """Prepare enhanced analysis data with additional context"""
        sections = extraction_data.get('extracted_sections', [])
        text_summary = extraction_data.get('text_summary', {})
        enhancement_features = extraction_data.get('enhancement_features', {})
        
        # Enhanced summary
        analysis_summary = []
        
        analysis_summary.append(f"ENHANCED EXTRACTION SUMMARY:")
        analysis_summary.append(f"- Total items: {len(sections)}")
        analysis_summary.append(f"- Text summary available: {bool(text_summary.get('summary'))}")
        analysis_summary.append(f"- Reference matching used: {enhancement_features.get('reference_matching', False)}")
        analysis_summary.append(f"- Computer vision available: {enhancement_features.get('cv_available', False)}")
        
        # Text summary analysis
        if text_summary.get('summary'):
            summary_text = text_summary['summary']
            analysis_summary.append(f"\nTEXT SUMMARY ANALYSIS:")
            analysis_summary.append(f"- Summary length: {len(summary_text)} characters")
            analysis_summary.append(f"- Source pages: {text_summary.get('source_pages', [])}")
            analysis_summary.append(f"- Relevant sentences found: {text_summary.get('sentence_count', 0)}")
            analysis_summary.append(f"- Summary confidence: {text_summary.get('confidence', 0):.3f}")
            
            # Include key parts of summary for context
            if len(summary_text) > 0:
                analysis_summary.append(f"\nKEY SUMMARY CONTENT:")
                summary_preview = summary_text[:500] + "..." if len(summary_text) > 500 else summary_text
                analysis_summary.append(f"'{summary_preview}'")
        
        # Detailed extraction analysis
        analysis_summary.append(f"\nDETAILED EXTRACTION RESULTS:")
        
        # Group by type
        type_groups = {}
        for section in sections:
            section_type = section.get('type', 'unknown')
            if section_type not in type_groups:
                type_groups[section_type] = []
            type_groups[section_type].append(section)
        
        for section_type, items in type_groups.items():
            analysis_summary.append(f"\n{section_type.upper()} ({len(items)} items):")
            
            for i, item in enumerate(items[:3], 1):  # Show top 3 per type
                confidence = item.get('confidence', 0)
                page = item.get('page', 'unknown')
                
                analysis_summary.append(f"{i}. Page {page}, Confidence: {confidence:.3f}")
                
                # Add specific metadata based on type
                if 'image' in section_type:
                    metadata = item.get('metadata', {})
                    method = metadata.get('extraction_method', 'unknown')
                    ref_match = metadata.get('reference_match', 0)
                    analysis_summary.append(f"   Method: {method}, Reference match: {ref_match:.3f}")
                
                elif 'table' in section_type:
                    table_analysis = item.get('table_analysis', {})
                    data_points = table_analysis.get('data_points_found', [])
                    if data_points:
                        analysis_summary.append(f"   Data points: {', '.join(data_points[:3])}")
                
                elif 'summary' in section_type:
                    metadata = item.get('metadata', {})
                    source_pages = metadata.get('source_pages', [])
                    analysis_summary.append(f"   Source pages: {source_pages}")
                
                # Content preview
                content = item.get('content', '')
                if len(content) > 100:
                    content_preview = content[:100] + "..."
                else:
                    content_preview = content
                analysis_summary.append(f"   Content: {content_preview}")
            
            if len(items) > 3:
                analysis_summary.append(f"   ... and {len(items) - 3} more {section_type} items")
        
        # Reference data context
        if self.reference_data.get('total_references', 0) > 0:
            analysis_summary.append(f"\nREFERENCE DATA AVAILABLE:")
            analysis_summary.append(f"- Reference images: {self.reference_data.get('total_references', 0)}")
            analysis_summary.append(f"- Reference types: {list(self.reference_data.get('descriptions', {}).keys())}")
        
        # Current parameters context
        params = extraction_data.get('parameters_used', {})
        analysis_summary.append(f"\nCURRENT PARAMETERS:")
        for param, value in params.items():
            analysis_summary.append(f"- {param}: {value}")
        
        return '\n'.join(analysis_summary)
    
    def apply_enhanced_recommendations(self, document_id: str, analysis_result: Dict, debug: bool = False) -> bool:
        """Apply enhanced LLM recommendations to parameters"""
        try:
            recommendations = analysis_result.get('parameter_recommendations', {})
            if not recommendations:
                print(f"ℹ️  No enhanced parameter recommendations for {self.characteristic}")
                return True
            
            # Load current parameters
            param_file = f"parameters_{self.characteristic}.json"
            try:
                if os.path.exists(param_file):
                    with open(param_file) as f:
                        current_params = json.load(f)
                else:
                    # Create default parameters structure
                    current_params = {
                        'confidence_threshold': 0.3,
                        'content_classification_threshold': 0.2,
                        'skip_pages': 3,
                        'image_size_min': 100,
                        'min_section_length': 100,
                        'max_extractions': 15,
                        'table_relevance_threshold': 1,
                        'text_summary_enabled': True,
                        'reference_matching_enabled': True
                    }
            except Exception as e:
                print(f"❌ Error loading current parameters: {e}")
                return False
            
            # Apply recommended changes with enhanced validation
            changes_made = []
            for param_name, new_value in recommendations.items():
                if param_name == 'reasoning':
                    continue  # Skip reasoning text
                
                if param_name in current_params and new_value is not None:
                    try:
                        # Enhanced type conversion and validation
                        if param_name in ['skip_pages', 'image_size_min', 'max_extractions', 'table_relevance_threshold', 'min_section_length']:
                            new_value = max(1, min(10 if param_name == 'skip_pages' else 1000, int(float(new_value))))
                        elif param_name in ['confidence_threshold', 'content_classification_threshold']:
                            new_value = max(0.05, min(0.95, float(new_value)))
                        else:
                            new_value = float(new_value)
                        
                        old_value = current_params[param_name]
                        
                        # Only apply if change is significant (enhanced threshold)
                        if abs(old_value - new_value) >= 0.005:
                            current_params[param_name] = new_value
                            changes_made.append(f"{param_name}: {old_value} → {new_value}")
                            
                    except (ValueError, TypeError) as e:
                        if debug:
                            print(f"⚠ Invalid enhanced recommendation for {param_name}: {new_value} ({e})")
                        continue
            
            # Save updated parameters
            if changes_made:
                with open(param_file, 'w') as f:
                    json.dump(current_params, f, indent=2)
                
                print(f"✅ Applied {len(changes_made)} enhanced parameter changes for {self.characteristic}:")
                for change in changes_made:
                    print(f"   • {change}")
                
                # Enhanced reasoning display
                reasoning = recommendations.get('reasoning', 'No reasoning provided')
                print(f"💡 Enhanced Reasoning: {reasoning}")
                
            else:
                print(f"ℹ️  No significant enhanced parameter changes needed for {self.characteristic}")
            
            # Save enhanced feedback log
            self._save_enhanced_feedback_log(document_id, analysis_result, changes_made)
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying enhanced recommendations: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _save_enhanced_feedback_log(self, document_id: str, analysis_result: Dict, changes_made: List[str]):
        """Save enhanced feedback analysis log"""
        quality = analysis_result.get('extraction_quality', {})
        enhanced_feedback = analysis_result.get('enhanced_feedback', {})
        recommendations = analysis_result.get('parameter_recommendations', {})
        
        log_entry = {
            'document_id': document_id,
            'characteristic': self.characteristic,
            'timestamp': datetime.now().isoformat(),
            'enhanced_quality_scores': {
                'relevance_accuracy': quality.get('relevance_accuracy', 0),
                'completeness': quality.get('completeness', 0),
                'precision': quality.get('precision', 0),
                'context_appropriateness': quality.get('context_appropriateness', 0),
                'average': sum([quality.get('relevance_accuracy', 0), 
                              quality.get('completeness', 0), 
                              quality.get('precision', 0),
                              quality.get('context_appropriateness', 0)]) / 4
            },
            'enhanced_analysis': {
                'correctly_identified': enhanced_feedback.get('correctly_identified', []),
                'incorrectly_classified': enhanced_feedback.get('incorrectly_classified', []),
                'missing_content': enhanced_feedback.get('missing_content', []),
                'reference_alignment': enhanced_feedback.get('reference_alignment', 'unknown')
            },
            'parameter_changes': changes_made,
            'enhanced_reasoning': recommendations.get('reasoning', ''),
            'reference_data_used': {
                'total_references': self.reference_data.get('total_references', 0),
                'reference_types': list(self.reference_data.get('descriptions', {}).keys())
            },
            'llm_available': True,
            'llm_actually_used': True,
            'analysis_type': 'enhanced_with_reference_data'
        }
        
        # Save to enhanced characteristic-specific log
        log_file = f"enhanced_feedback_log_{self.characteristic}.json"
        try:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep last 25 entries (enhanced logs are more detailed)
            if len(logs) > 25:
                logs = logs[-25:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            print(f"📝 Enhanced feedback logged to {log_file}")
            
        except Exception as e:
            print(f"⚠ Error saving enhanced feedback log: {e}")

def test_enhanced_connection():
    """Test enhanced Azure OpenAI connection"""
    print("🔧 Testing Enhanced Azure OpenAI connection...")
    
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    if not LANGCHAIN_OK:
        print("❌ LangChain not available - install with: pip install langchain-openai")
        return False
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version="2024-02-01",
            temperature=0.1,
            max_tokens=100
        )
        
        # Test with enhanced message
        messages = [HumanMessage(content="Reply with 'Enhanced connection successful' if you can read this and have access to analyze construction documents.")]
        response = llm.invoke(messages)
        
        if "successful" in response.content.lower():
            print("✅ Enhanced Azure OpenAI connection successful!")
            print(f"✓ Enhanced endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            print(f"✓ Enhanced deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
            return True
        else:
            print(f"⚠ Unexpected response: {response.content}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced connection failed: {e}")
        return False

def show_enhanced_logs(characteristic: str = None):
    """Show enhanced feedback logs"""
    if characteristic:
        log_files = [f"enhanced_feedback_log_{characteristic}.json"]
        print(f"📋 Showing Enhanced {characteristic.upper()} feedback logs:")
    else:
        # Show all enhanced characteristic logs
        characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
        log_files = [f"enhanced_feedback_log_{char}.json" for char in characteristics]
        print("📋 Showing all enhanced characteristic feedback logs:")
    
    found_logs = False
    
    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs = True
            try:
                with open(log_file) as f:
                    logs = json.load(f)
                
                if not logs:
                    continue
                
                char_name = log_file.replace('enhanced_feedback_log_', '').replace('.json', '')
                print(f"\n{'='*60}")
                print(f"🎯 ENHANCED {char_name.upper()} FEEDBACK HISTORY")
                print(f"{'='*60}")
                
                # Show last 3 entries with enhanced details
                recent_logs = logs[-3:]
                for i, log in enumerate(reversed(recent_logs), 1):
                    timestamp = datetime.fromisoformat(log.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
                    doc_id = log.get('document_id', 'unknown')
                    
                    quality = log.get('enhanced_quality_scores', {})
                    avg_quality = quality.get('average', 0)
                    
                    changes = log.get('parameter_changes', [])
                    change_count = len(changes)
                    
                    ref_data = log.get('reference_data_used', {})
                    ref_count = ref_data.get('total_references', 0)
                    
                    print(f"\n{i}. {timestamp} | Doc: {doc_id[:8]}...")
                    print(f"   Quality: {avg_quality:.1f}/5 | Changes: {change_count} | Refs: {ref_count}")
                    
                    enhanced_analysis = log.get('enhanced_analysis', {})
                    correctly_id = enhanced_analysis.get('correctly_identified', [])
                    incorrectly_cl = enhanced_analysis.get('incorrectly_classified', [])
                    
                    if correctly_id:
                        print(f"   ✅ Correct: {len(correctly_id)} items")
                    if incorrectly_cl:
                        print(f"   ❌ Incorrect: {len(incorrectly_cl)} items")
                    
                    if changes:
                        print(f"   Updates: {', '.join(changes[:2])}{'...' if len(changes) > 2 else ''}")
                    
                    reasoning = log.get('enhanced_reasoning', '')
                    if reasoning and len(reasoning) > 0:
                        print(f"   Reason: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
                
                print(f"\n📈 Total enhanced {char_name} feedback entries: {len(logs)}")
                
            except Exception as e:
                print(f"❌ Error reading enhanced {log_file}: {e}")
    
    if not found_logs:
        print("ℹ️  No enhanced feedback logs found. Run some enhanced extractions first!")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Characteristic-Specific LLM Feedback Analyzer")
    parser.add_argument("--enhanced-analyze", help="Analyze enhanced characteristic extraction")
    parser.add_argument("--source-pdf", help="Path to original PDF for enhanced analysis")
    parser.add_argument("--test-connection", action="store_true", help="Test enhanced Azure OpenAI connection")
    parser.add_argument("--show-log", action="store_true", help="Show enhanced feedback logs")
    parser.add_argument("--characteristic", help="Filter logs by characteristic")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("document_id", nargs="?", help="Document ID to analyze")
    
    args = parser.parse_args()
    
    if args.test_connection:
        success = test_enhanced_connection()
        return 0 if success else 1
    
    if args.show_log:
        show_enhanced_logs(args.characteristic)
        return 0
    
    if args.enhanced_analyze and args.document_id:
        characteristic = args.enhanced_analyze
        document_id = args.document_id
        source_pdf = args.source_pdf
        
        print(f"🤖 Starting Enhanced {characteristic.upper()} LLM Analysis")
        print(f"📄 Document ID: {document_id}")
        print(f"📋 Original PDF: {os.path.basename(source_pdf) if source_pdf else 'Not provided'}")
        
        if not LANGCHAIN_OK:
            print("❌ LangChain not available")
            return 1
        
        try:
            analyzer = EnhancedCharacteristicLLMAnalyzer(characteristic)
            
            # Enhanced analysis
            print(f"🔍 Analyzing enhanced {characteristic} extraction...")
            analysis_result = analyzer.analyze_enhanced_extraction(document_id, source_pdf, args.debug)
            
            if not analysis_result:
                print(f"❌ Failed to analyze enhanced {characteristic} extraction")
                return 1
            
            print(f"✅ Enhanced analysis completed for {characteristic}")
            
            # Show enhanced quality scores
            quality = analysis_result.get('extraction_quality', {})
            print(f"\n📊 ENHANCED {characteristic.upper()} QUALITY SCORES:")
            print(f"   Relevance Accuracy: {quality.get('relevance_accuracy', 0)}/5")
            print(f"   Completeness: {quality.get('completeness', 0)}/5") 
            print(f"   Precision: {quality.get('precision', 0)}/5")
            print(f"   Context Appropriateness: {quality.get('context_appropriateness', 0)}/5")
            
            # Apply enhanced recommendations
            print(f"\n⚙️  Applying enhanced {characteristic} recommendations...")
            success = analyzer.apply_enhanced_recommendations(document_id, analysis_result, args.debug)
            
            if success:
                print(f"✅ Enhanced {characteristic.title()} feedback processing completed")
                return 0
            else:
                print(f"⚠️  Some issues applying enhanced {characteristic} recommendations")
                return 1
                
        except Exception as e:
            print(f"❌ Enhanced {characteristic.title()} analysis error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    else:
        print("❌ Missing required arguments for enhanced analysis")
        print("Usage examples:")
        print("  python llm_feedback.py --test-connection")
        print("  python llm_feedback.py --enhanced-analyze anchors doc123 --source-pdf document.pdf")
        print("  python llm_feedback.py --show-log --characteristic glazing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
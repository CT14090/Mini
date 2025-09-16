#!/usr/bin/env python3
"""
Enhanced Window Characteristic LLM Feedback System
Provides AI-powered analysis with access to original PDF, reference data, and extracted content
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
except:
    pass

# LangChain imports
LANGCHAIN_OK = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_OK = True
    print("✓ LangChain Azure OpenAI available for enhanced feedback")
except ImportError as e:
    print(f"⚠ LangChain not available: {e}")

class WindowCharacteristicLLMAnalyzer:
    """Enhanced LLM analyzer for window characteristics with full context access"""
    
    def __init__(self, characteristic: str):
        self.characteristic = characteristic
        
        # Force reload environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            print("✅ Environment variables reloaded")
        except Exception as e:
            print(f"⚠️ Error loading environment: {e}")
        
        # Check if LangChain is available
        if not LANGCHAIN_OK:
            print("❌ LangChain is required for LLM feedback!")
            print("   Install with: pip install langchain-openai langchain-core")
            raise RuntimeError("LangChain required but not available")
        
        # Check Azure OpenAI configuration with detailed validation
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        config_status = {}
        
        for var in required_vars:
            value = os.getenv(var)
            config_status[var] = value is not None and len(str(value).strip()) > 0
            if config_status[var]:
                if 'KEY' in var:
                    print(f"✅ {var}: {str(value)[:8]}...{str(value)[-4:]}")
                else:
                    print(f"✅ {var}: {value}")
            else:
                print(f"❌ {var}: Not set or empty")
        
        missing_vars = [var for var, status in config_status.items() if not status]
        
        if missing_vars:
            print(f"❌ Azure OpenAI configuration incomplete: {missing_vars}")
            print("   Add these to your .env file:")
            for var in missing_vars:
                print(f"   {var}=your_value_here")
            print("   Example .env:")
            print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("   AZURE_OPENAI_API_KEY=your-32-character-key")
            print("   AZURE_OPENAI_DEPLOYMENT=your-gpt-4-deployment-name")
            raise RuntimeError("Azure OpenAI configuration required but incomplete")
        
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=3000
            )
            
            # Test the connection immediately
            print(f"🧪 Testing Azure OpenAI connection for {characteristic}...")
            test_response = self.llm.invoke([HumanMessage(content="Reply 'OK' if you can analyze window construction documents.")])
            
            if test_response and test_response.content and 'OK' in test_response.content:
                print(f"✅ Azure OpenAI connection verified for {characteristic} analysis")
            else:
                print(f"⚠️ Unexpected test response: {test_response.content if test_response else 'None'}")
                
        except Exception as e:
            print(f"❌ Azure OpenAI initialization failed: {e}")
            print("   Check your credentials and try running:")
            print("   python azure_openai_checker.py")
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {e}")
        
        # Load reference data and context
        self.reference_data = self._load_reference_data()
        self.analysis_prompts = self._get_analysis_prompts()
        
        print(f"🎯 Enhanced LLM analyzer ready for {characteristic} with {self.reference_data.get('total_references', 0)} references")
    
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
        image_files = list(reference_dir.glob("*.jpg")) + list(reference_dir.glob("*.png")) + list(reference_dir.glob("*.jpeg"))
        
        reference_context = self._create_reference_context(descriptions, image_files)
        
        print(f"✓ Loaded reference data: {len(descriptions)} descriptions, {len(image_files)} images")
        
        return {
            'descriptions': descriptions,
            'images': [str(f.name) for f in image_files],
            'context': reference_context,
            'total_references': len(image_files) + len(descriptions)
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
    
    def _get_analysis_prompts(self) -> Dict[str, str]:
        """Get analysis prompts with reference data context for each characteristic"""
        base_context = f"""
REFERENCE DATA CONTEXT:
{self.reference_data.get('context', 'No reference data available')}

You are analyzing window {self.characteristic.replace('_', ' ')} extraction results with full access to:
1. Original PDF document context
2. Reference training data (descriptions and images)
3. Extracted content (text summaries, images, tables)
4. Current extraction parameters

Your goal is to evaluate extraction quality and recommend parameter improvements.
"""
        
        return {
            'anchors': base_context + """
ANCHOR TYPES TO IDENTIFY:
- Directly Into Concrete: Anchors/screws going directly into concrete substrate
- Directly Into Wood: Screws/fasteners going directly into wood framing
- Into Wood via 1By Buck: Anchors going through 1x wood buck into substrate
- Into Concrete via 1By Buck: Anchors going through 1x wood buck into concrete
- Into Concrete via 2By Buck: Anchors going through 2x wood buck into concrete
- Self Drilling Screws Into Metal Structures: Self-drilling screws for metal framing

EVALUATION FOCUS:
1. Were the correct anchor types identified based on text content and visual elements?
2. Do extracted images actually show anchor details, connection points, or fastening systems?
3. Do extracted tables contain anchor specifications, fastener schedules, or connection data?
4. Was irrelevant content (general construction details, logos) properly filtered out?
5. Does the text summary accurately capture anchor-related information?

PARAMETER OPTIMIZATION:
Focus on precision over recall - it's better to miss some anchor details than extract 
irrelevant content. Recommend threshold adjustments that improve classification accuracy.
""",
            
            'glazing': base_context + """
GLAZING CHARACTERISTICS TO IDENTIFY:
- Glass type and thickness specifications (single, double, triple glazing)
- Low-E coatings and performance ratings
- Insulated glass unit (IGU/IGP) configurations and details
- Laminated vs tempered vs annealed glass specifications
- Thermal and optical properties (U-values, SHGC, VLT)
- Glazing system assembly details and installation

EVALUATION FOCUS:
1. Were glass specifications correctly identified from text and visual content?
2. Do extracted images show actual glazing details, cross-sections, or assemblies?
3. Are glazing performance data and ratings properly captured?
4. Does the text summary focus on glazing-specific information?
5. Was general window content mistaken for glazing-specific details?

PARAMETER OPTIMIZATION:
Optimize for glazing-specific content recognition while filtering out general window information.
""",
            
            'impact_rating': base_context + """
IMPACT RATING TYPES TO IDENTIFY:
- Small Missile Impact ratings and test results
- Large Missile Impact ratings and test results  
- Both Missile Impact comprehensive ratings
- ASTM E1886/E1996 test compliance documentation
- AAMA/WDMA test standards and certifications
- Hurricane/storm impact ratings and approvals

EVALUATION FOCUS:
1. Were impact ratings correctly identified and properly categorized?
2. Do extracted images show test certifications, impact data, or compliance documents?
3. Are test standards and rating values properly captured?
4. Does the text summary focus on impact resistance information?
5. Was general performance data mistaken for impact-specific ratings?

PARAMETER OPTIMIZATION:
Focus on identifying actual impact test results and certifications, not general performance data.
""",
            
            'design_pressure': base_context + """
DESIGN PRESSURE DATA TO IDENTIFY:
- Design pressure ratings (DP values) in various units (PSF, PA, KPA)
- Positive and negative pressure ratings and specifications
- Wind load specifications and structural performance data
- Pressure test results and compliance documentation
- Structural load ratings and deflection limits
- Performance class ratings and certifications

EVALUATION FOCUS:
1. Were design pressure values and ratings accurately extracted?
2. Do extracted tables contain actual pressure data, load specifications, or performance metrics?
3. Are pressure units and rating systems properly identified?
4. Does the text summary capture pressure-related technical information?
5. Was general structural data mistaken for pressure-specific information?

PARAMETER OPTIMIZATION:
Optimize for numerical pressure data and technical specifications while filtering general content.
"""
        }
    
    def analyze_extraction(self, document_id: str, source_pdf: str = None, debug: bool = False) -> Optional[Dict]:
        """Analyze extraction with comprehensive context"""
        try:
            # Load extraction data
            extraction_file = f"feedback_data/{self.characteristic}_extraction_{document_id}.json"
            if not os.path.exists(extraction_file):
                print(f"❌ Extraction file not found: {extraction_file}")
                return None
            
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            print(f"📊 Analyzing {self.characteristic} extraction: {len(extraction_data.get('extracted_sections', []))} items")
            
            # Prepare comprehensive analysis data
            analysis_data = self._prepare_comprehensive_analysis(extraction_data, source_pdf, debug)
            
            # Create system and user prompts
            system_prompt = self.analysis_prompts.get(self.characteristic, self.analysis_prompts['anchors'])
            
            user_prompt = f"""
COMPREHENSIVE {self.characteristic.upper()} EXTRACTION ANALYSIS

DOCUMENT CONTEXT:
- Original PDF: {os.path.basename(source_pdf) if source_pdf else 'Not provided'}
- Document ID: {document_id}
- Total pages processed: {extraction_data.get('total_pages', 'unknown')}
- Processing method: {extraction_data.get('processing_method', 'unknown')}

EXTRACTION SUMMARY:
{analysis_data}

REFERENCE DATA ALIGNMENT:
The extraction should align with the reference descriptions and examples provided above.
Compare extracted content against these reference standards for accuracy assessment.

CRITICAL EVALUATION REQUIRED:

1. CONTENT RELEVANCE (Score 1-5):
   - Are extracted text summaries actually about {self.characteristic.replace('_', ' ')}?
   - Do extracted images show {self.characteristic.replace('_', ' ')}-related content?
   - Do extracted tables contain {self.characteristic.replace('_', ' ')} specifications or data?

2. EXTRACTION QUALITY (Score 1-5):
   - Accuracy: Were the right items identified?
   - Completeness: Was important {self.characteristic.replace('_', ' ')} content captured?
   - Precision: Were false positives minimized?
   - Context: Does extraction make sense for this document type?

3. REFERENCE ALIGNMENT (Score 1-5):
   - How well do extracted items match the reference descriptions?
   - Are there obvious misclassifications based on reference standards?
   - Does the content type align with expected {self.characteristic.replace('_', ' ')} characteristics?

4. PARAMETER OPTIMIZATION:
   Current parameters: {json.dumps(extraction_data.get('parameters_used', {}), indent=2)}
   
   Recommend specific adjustments for:
   - confidence_threshold (currently {extraction_data.get('parameters_used', {}).get('confidence_threshold', 0.3)})
   - content_classification_threshold (currently {extraction_data.get('parameters_used', {}).get('content_classification_threshold', 0.2)})
   - skip_pages (currently {extraction_data.get('parameters_used', {}).get('skip_pages', 3)})
   - image_size_min (currently {extraction_data.get('parameters_used', {}).get('image_size_min', 100)})
   - max_extractions (currently {extraction_data.get('parameters_used', {}).get('max_extractions', 15)})

RESPOND IN JSON FORMAT:
{{
    "analysis_summary": {{
        "content_relevance_score": <1-5>,
        "extraction_quality_score": <1-5>,
        "reference_alignment_score": <1-5>,
        "overall_assessment": "<detailed assessment>"
    }},
    "content_evaluation": {{
        "correctly_identified": ["list of correctly identified items"],
        "incorrectly_classified": ["list of items that don't match {self.characteristic}"],
        "missing_content_likely": ["types of {self.characteristic} content likely missed"],
        "extraction_strengths": ["what the extraction did well"],
        "extraction_weaknesses": ["areas needing improvement"]
    }},
    "parameter_recommendations": {{
        "confidence_threshold": <recommended_value_or_null>,
        "content_classification_threshold": <recommended_value_or_null>,
        "skip_pages": <recommended_value_or_null>,
        "image_size_min": <recommended_value_or_null>,
        "max_extractions": <recommended_value_or_null>,
        "detailed_reasoning": "<explain why these changes will improve {self.characteristic} extraction>"
    }}
}}
"""
            
            if debug:
                print(f"🤖 Sending {self.characteristic} analysis to Azure OpenAI...")
                print(f"Analysis data length: {len(analysis_data)} characters")
            
            # Get LLM analysis
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if debug:
                print(f"✓ LLM response received: {len(response.content)} characters")
            
            # Parse JSON response
            try:
                # Clean the response - remove markdown code blocks if present
                response_content = response.content.strip()
                if response_content.startswith('```json'):
                    response_content = response_content.replace('```json', '').replace('```', '').strip()
                elif response_content.startswith('```'):
                    response_content = response_content.replace('```', '').strip()
                
                analysis_result = json.loads(response_content)
                
                if debug:
                    print(f"✓ Successfully parsed LLM analysis for {self.characteristic}")
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM response as JSON: {e}")
                if debug:
                    print(f"Raw response: {response.content[:500]}...")
                return None
                
        except Exception as e:
            print(f"❌ LLM analysis error: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _prepare_comprehensive_analysis(self, extraction_data: Dict, source_pdf: str, debug: bool = False) -> str:
        """Prepare comprehensive analysis data with full context"""
        sections = extraction_data.get('extracted_sections', [])
        text_summary = extraction_data.get('text_summary', {})
        enhancement_features = extraction_data.get('enhancement_features', {})
        
        analysis_parts = []
        
        # Overall extraction summary
        analysis_parts.append(f"EXTRACTION OVERVIEW:")
        analysis_parts.append(f"- Total extracted items: {len(sections)}")
        analysis_parts.append(f"- Text summary enabled: {enhancement_features.get('text_summary_enabled', False)}")
        analysis_parts.append(f"- Reference matching enabled: {enhancement_features.get('reference_matching', False)}")
        analysis_parts.append(f"- Computer vision available: {enhancement_features.get('cv_available', False)}")
        analysis_parts.append(f"- Pages skipped: {enhancement_features.get('pages_skipped', 0)}")
        
        # Text summary analysis
        if text_summary.get('summary'):
            analysis_parts.append(f"\nTEXT SUMMARY ANALYSIS:")
            summary_text = text_summary['summary']
            analysis_parts.append(f"- Summary length: {len(summary_text)} characters")
            analysis_parts.append(f"- Source pages: {text_summary.get('source_pages', [])}")
            analysis_parts.append(f"- Sentence count: {text_summary.get('sentence_count', 0)}")
            analysis_parts.append(f"- Summary confidence: {text_summary.get('confidence', 0):.3f}")
            
            # Include preview of summary content
            if len(summary_text) > 0:
                summary_preview = summary_text[:400] + "..." if len(summary_text) > 400 else summary_text
                analysis_parts.append(f"- Content preview: '{summary_preview}'")
        else:
            analysis_parts.append(f"\nTEXT SUMMARY: None extracted")
        
        # Detailed item-by-item analysis
        analysis_parts.append(f"\nDETAILED ITEM ANALYSIS:")
        
        # Group items by type
        type_groups = {}
        for section in sections:
            section_type = section.get('type', 'unknown')
            if section_type not in type_groups:
                type_groups[section_type] = []
            type_groups[section_type].append(section)
        
        for section_type, items in type_groups.items():
            analysis_parts.append(f"\n{section_type.upper()} ITEMS ({len(items)}):")
            
            # Show details for each item (limit to first 3 per type for brevity)
            for i, item in enumerate(items[:3], 1):
                confidence = item.get('confidence', 0)
                page = item.get('page', 'unknown')
                extraction_id = item.get('extraction_id', 'unknown')
                
                analysis_parts.append(f"  {i}. ID: {extraction_id}")
                analysis_parts.append(f"     Page: {page} | Confidence: {confidence:.3f}")
                
                # Type-specific metadata
                metadata = item.get('metadata', {})
                if 'image' in section_type:
                    method = metadata.get('extraction_method', 'unknown')
                    ref_match = metadata.get('reference_match', 0)
                    ref_name = metadata.get('reference_name', 'none')
                    analysis_parts.append(f"     Method: {method} | Ref match: {ref_match:.3f} | Ref: {ref_name}")
                
                elif 'table' in section_type:
                    relevance = metadata.get('relevance_score', 0)
                    table_analysis = metadata.get('table_analysis', {})
                    data_points = table_analysis.get('data_points_found', [])
                    analysis_parts.append(f"     Relevance: {relevance:.1f} | Data points: {data_points[:3]}")
                
                elif 'summary' in section_type:
                    keywords = metadata.get('keyword_matches', 0)
                    sentences = metadata.get('sentence_count', 0)
                    analysis_parts.append(f"     Keywords: {keywords} | Sentences: {sentences}")
                
                # Content preview
                content = item.get('content', '')
                if len(content) > 150:
                    content_preview = content[:150] + "..."
                else:
                    content_preview = content
                analysis_parts.append(f"     Content: '{content_preview}'")
            
            if len(items) > 3:
                analysis_parts.append(f"     ... and {len(items) - 3} more {section_type} items")
        
        # Reference data usage
        ref_data_used = enhancement_features.get('reference_data_used', {})
        analysis_parts.append(f"\nREFERENCE DATA USAGE:")
        analysis_parts.append(f"- Total references available: {ref_data_used.get('total_references', 0)}")
        analysis_parts.append(f"- Image references: {ref_data_used.get('image_references', 0)}")
        analysis_parts.append(f"- Description references: {ref_data_used.get('description_references', 0)}")
        
        # Current parameter context
        params = extraction_data.get('parameters_used', {})
        analysis_parts.append(f"\nCURRENT PARAMETERS:")
        for param, value in params.items():
            analysis_parts.append(f"- {param}: {value}")
        
        return '\n'.join(analysis_parts)
    
    def apply_recommendations(self, document_id: str, analysis_result: Dict, debug: bool = False) -> bool:
        """Apply LLM recommendations to characteristic parameters"""
        try:
            recommendations = analysis_result.get('parameter_recommendations', {})
            if not recommendations:
                print(f"ℹ️  No parameter recommendations for {self.characteristic}")
                return True
            
            # Load current parameters
            param_file = f"parameters_{self.characteristic}.json"
            try:
                if os.path.exists(param_file):
                    with open(param_file) as f:
                        current_params = json.load(f)
                else:
                    # Create default parameters
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
            
            # Apply recommended changes with validation
            changes_made = []
            for param_name, new_value in recommendations.items():
                if param_name == 'detailed_reasoning':
                    continue  # Skip reasoning text
                
                if param_name in current_params and new_value is not None:
                    try:
                        # Type conversion and bounds checking
                        if param_name in ['skip_pages', 'image_size_min', 'max_extractions', 'table_relevance_threshold', 'min_section_length']:
                            if param_name == 'skip_pages':
                                new_value = max(0, min(10, int(float(new_value))))
                            elif param_name == 'max_extractions':
                                new_value = max(5, min(50, int(float(new_value))))
                            else:
                                new_value = max(50, min(500, int(float(new_value))))
                        
                        elif param_name in ['confidence_threshold', 'content_classification_threshold']:
                            new_value = max(0.05, min(0.95, float(new_value)))
                        else:
                            new_value = float(new_value)
                        
                        old_value = current_params[param_name]
                        
                        # Only apply if change is significant
                        if isinstance(new_value, int):
                            threshold = 1
                        else:
                            threshold = 0.01
                        
                        if abs(old_value - new_value) >= threshold:
                            current_params[param_name] = new_value
                            changes_made.append(f"{param_name}: {old_value} → {new_value}")
                            
                    except (ValueError, TypeError) as e:
                        if debug:
                            print(f"⚠ Invalid recommendation for {param_name}: {new_value} ({e})")
                        continue
            
            # Save updated parameters
            if changes_made:
                with open(param_file, 'w') as f:
                    json.dump(current_params, f, indent=2)
                
                print(f"✅ Applied {len(changes_made)} parameter changes for {self.characteristic}:")
                for change in changes_made:
                    print(f"   • {change}")
                
                # Display reasoning
                reasoning = recommendations.get('detailed_reasoning', 'No reasoning provided')
                print(f"💡 Reasoning: {reasoning}")
                
            else:
                print(f"ℹ️  No significant parameter changes needed for {self.characteristic}")
            
            # Save feedback log
            self._save_feedback_log(document_id, analysis_result, changes_made)
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying recommendations: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _save_feedback_log(self, document_id: str, analysis_result: Dict, changes_made: List[str]):
        """Save feedback analysis log"""
        analysis_summary = analysis_result.get('analysis_summary', {})
        content_eval = analysis_result.get('content_evaluation', {})
        recommendations = analysis_result.get('parameter_recommendations', {})
        
        log_entry = {
            'document_id': document_id,
            'characteristic': self.characteristic,
            'timestamp': datetime.now().isoformat(),
            'quality_scores': {
                'content_relevance': analysis_summary.get('content_relevance_score', 0),
                'extraction_quality': analysis_summary.get('extraction_quality_score', 0),
                'reference_alignment': analysis_summary.get('reference_alignment_score', 0),
                'overall_assessment': analysis_summary.get('overall_assessment', 'No assessment provided')
            },
            'content_analysis': {
                'correctly_identified': content_eval.get('correctly_identified', []),
                'incorrectly_classified': content_eval.get('incorrectly_classified', []),
                'missing_content': content_eval.get('missing_content_likely', []),
                'strengths': content_eval.get('extraction_strengths', []),
                'weaknesses': content_eval.get('extraction_weaknesses', [])
            },
            'parameter_changes': changes_made,
            'reasoning': recommendations.get('detailed_reasoning', ''),
            'reference_data_used': {
                'total_references': self.reference_data.get('total_references', 0),
                'has_descriptions': bool(self.reference_data.get('descriptions')),
                'has_images': bool(self.reference_data.get('images'))
            },
            'llm_available': True,
            'llm_actually_used': True,
            'analysis_type': 'comprehensive_characteristic_analysis'
        }
        
        # Save to characteristic-specific log
        log_file = f"feedback_log_{self.characteristic}.json"
        try:
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # Keep last 30 entries
            if len(logs) > 30:
                logs = logs[-30:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            if len(changes_made) > 0:
                print(f"📝 Feedback logged with {len(changes_made)} parameter changes")
            else:
                print(f"📝 Feedback logged - no changes needed")
            
        except Exception as e:
            print(f"⚠ Error saving feedback log: {e}")

def test_connection():
    """Test Azure OpenAI connection"""
    print("🔧 Testing Azure OpenAI connection...")
    
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Add these to your .env file:")
        for var in missing_vars:
            print(f"   {var}=your_value_here")
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
        
        # Test message
        messages = [HumanMessage(content="Reply with 'Connection successful' if you can read this and are ready to analyze window construction documents.")]
        response = llm.invoke(messages)
        
        if "successful" in response.content.lower():
            print("✅ Azure OpenAI connection successful!")
            print(f"✓ Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            print(f"✓ Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
            return True
        else:
            print(f"⚠ Unexpected response: {response.content}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def show_logs(characteristic: str = None):
    """Show feedback logs"""
    if characteristic:
        log_files = [f"feedback_log_{characteristic}.json"]
        print(f"📋 Showing {characteristic.upper()} feedback logs:")
    else:
        # Show all characteristic logs
        characteristics = ['anchors', 'glazing', 'impact_rating', 'design_pressure']
        log_files = [f"feedback_log_{char}.json" for char in characteristics]
        print("📋 Showing all characteristic feedback logs:")
    
    found_logs = False
    
    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs = True
            try:
                with open(log_file) as f:
                    logs = json.load(f)
                
                if not logs:
                    continue
                
                char_name = log_file.replace('feedback_log_', '').replace('.json', '')
                print(f"\n{'='*60}")
                print(f"🎯 {char_name.upper()} FEEDBACK HISTORY")
                print(f"{'='*60}")
                
                # Show last 3 entries
                recent_logs = logs[-3:]
                for i, log in enumerate(reversed(recent_logs), 1):
                    timestamp = datetime.fromisoformat(log.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
                    doc_id = log.get('document_id', 'unknown')
                    
                    quality = log.get('quality_scores', {})
                    relevance = quality.get('content_relevance', 0)
                    extraction_q = quality.get('extraction_quality', 0)
                    alignment = quality.get('reference_alignment', 0)
                    
                    changes = log.get('parameter_changes', [])
                    change_count = len(changes)
                    
                    print(f"\n{i}. {timestamp} | Doc: {doc_id[:8]}...")
                    print(f"   Relevance: {relevance}/5 | Quality: {extraction_q}/5 | Alignment: {alignment}/5")
                    print(f"   Parameter changes: {change_count}")
                    
                    content_analysis = log.get('content_analysis', {})
                    correct = content_analysis.get('correctly_identified', [])
                    incorrect = content_analysis.get('incorrectly_classified', [])
                    
                    if correct:
                        print(f"   ✅ Correctly identified: {len(correct)} items")
                    if incorrect:
                        print(f"   ❌ Incorrectly classified: {len(incorrect)} items")
                    
                    if changes:
                        print(f"   🔧 Changes: {', '.join(changes[:2])}{'...' if len(changes) > 2 else ''}")
                    
                    reasoning = log.get('reasoning', '')
                    if reasoning and len(reasoning) > 0:
                        print(f"   💡 Reason: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
                
                print(f"\n📈 Total {char_name} feedback entries: {len(logs)}")
                
            except Exception as e:
                print(f"❌ Error reading {log_file}: {e}")
    
    if not found_logs:
        print("ℹ️  No feedback logs found. Process some documents first!")
        print("   Usage: python adaptive_agent.py --source document.pdf")

def main():
    parser = argparse.ArgumentParser(description="Window Characteristic LLM Feedback Analyzer")
    parser.add_argument("--enhanced-analyze", help="Analyze characteristic extraction (anchors, glazing, impact_rating, design_pressure)")
    parser.add_argument("--source-pdf", help="Path to original PDF for analysis")
    parser.add_argument("--test-connection", action="store_true", help="Test Azure OpenAI connection")
    parser.add_argument("--show-log", action="store_true", help="Show feedback logs")
    parser.add_argument("--characteristic", help="Filter logs by characteristic")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("document_id", nargs="?", help="Document ID to analyze")
    
    args = parser.parse_args()
    
    if args.test_connection:
        success = test_connection()
        return 0 if success else 1
    
    if args.show_log:
        show_logs(args.characteristic)
        return 0
    
    if args.enhanced_analyze and args.document_id:
        characteristic = args.enhanced_analyze
        document_id = args.document_id
        source_pdf = args.source_pdf
        
        if characteristic not in ['anchors', 'glazing', 'impact_rating', 'design_pressure']:
            print(f"❌ Unknown characteristic: {characteristic}")
            print("   Valid options: anchors, glazing, impact_rating, design_pressure")
            return 1
        
        print(f"🤖 Starting {characteristic.upper()} LLM Analysis")
        print(f"📄 Document ID: {document_id}")
        print(f"📋 Original PDF: {os.path.basename(source_pdf) if source_pdf else 'Not provided'}")
        
        if not LANGCHAIN_OK:
            print("❌ LangChain not available")
            return 1
        
        try:
            analyzer = WindowCharacteristicLLMAnalyzer(characteristic)
            
            # Run analysis
            print(f"🔍 Analyzing {characteristic} extraction...")
            analysis_result = analyzer.analyze_extraction(document_id, source_pdf, args.debug)
            
            if not analysis_result:
                print(f"❌ Failed to analyze {characteristic} extraction")
                return 1
            
            print(f"✅ Analysis completed for {characteristic}")
            
            # Show quality scores
            analysis_summary = analysis_result.get('analysis_summary', {})
            print(f"\n📊 {characteristic.upper()} QUALITY SCORES:")
            print(f"   Content Relevance: {analysis_summary.get('content_relevance_score', 0)}/5")
            print(f"   Extraction Quality: {analysis_summary.get('extraction_quality_score', 0)}/5")
            print(f"   Reference Alignment: {analysis_summary.get('reference_alignment_score', 0)}/5")
            
            overall_assessment = analysis_summary.get('overall_assessment', '')
            if overall_assessment:
                print(f"   Overall: {overall_assessment}")
            
            # Apply recommendations
            print(f"\n⚙️  Applying {characteristic} recommendations...")
            success = analyzer.apply_recommendations(document_id, analysis_result, args.debug)
            
            if success:
                print(f"✅ {characteristic.title()} feedback processing completed")
                return 0
            else:
                print(f"⚠️  Some issues applying {characteristic} recommendations")
                return 1
                
        except Exception as e:
            print(f"❌ {characteristic.title()} analysis error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    else:
        print("❌ Missing required arguments")
        print("Usage examples:")
        print("  python llm_feedback.py --test-connection")
        print("  python llm_feedback.py --enhanced-analyze anchors doc123 --source-pdf document.pdf")
        print("  python llm_feedback.py --show-log --characteristic glazing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
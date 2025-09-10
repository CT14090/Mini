#!/usr/bin/env python3
"""
Characteristic-Specific LLM Feedback Analyzer for Window Extraction
Evaluates extraction quality for specific window characteristics
"""
import json
import os
import sys
import pathlib
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# LangChain imports
LANGCHAIN_OK = False
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_OK = True
    print("✓ LangChain Azure OpenAI available")
except ImportError as e:
    print(f"⚠ LangChain not available: {e}")

class CharacteristicLLMAnalyzer:
    """LLM analyzer for window characteristic extractions"""
    
    def __init__(self, characteristic: str):
        self.characteristic = characteristic
        self.llm = None
        
        if LANGCHAIN_OK:
            try:
                self.llm = AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    api_version="2024-02-01",
                    temperature=0.1,
                    max_tokens=1500
                )
                print(f"✓ Azure OpenAI initialized for {characteristic} analysis")
            except Exception as e:
                print(f"❌ Azure OpenAI setup failed: {e}")
                self.llm = None
        
        # Characteristic-specific analysis prompts
        self.analysis_prompts = self._get_analysis_prompts()
    
    def _get_analysis_prompts(self) -> Dict[str, str]:
        """Get characteristic-specific analysis prompts"""
        return {
            'anchors': """
You are analyzing window anchor extraction results. Focus on:

ANCHOR TYPES TO IDENTIFY:
- Directly Into Concrete
- Directly Into Wood  
- Into Wood via 1By Buck
- Into Concrete via 1By Buck
- Into Concrete via 2By Buck
- Self Drilling Screws Into Metal Structures

EVALUATION CRITERIA:
1. Were anchor types correctly identified from images/tables/text?
2. Are anchor specifications (sizes, materials, installation methods) captured?
3. Are installation details and fastener specifications present?
4. Is the extraction complete for all anchor information in the document?

PARAMETER RECOMMENDATIONS:
- Adjust confidence_threshold if anchor details are missed or over-extracted
- Modify image_size_min if anchor detail images are too small/large
- Change table_relevance_threshold if anchor tables are missed
- Tune content_classification_threshold for better anchor recognition
""",
            
            'glazing': """
You are analyzing window glazing extraction results. Focus on:

GLAZING CHARACTERISTICS TO IDENTIFY:
- Glass type and thickness specifications
- Low-E coatings and performance ratings
- Insulated glass unit (IGU/IGP) configurations
- Laminated vs tempered vs annealed glass
- Double/triple glazing specifications
- Thermal and optical properties

EVALUATION CRITERIA:
1. Are glass types and thicknesses accurately extracted?
2. Are glazing specifications captured from technical tables?
3. Are glazing performance data and ratings identified?
4. Is the extraction comprehensive for all glazing information?

PARAMETER RECOMMENDATIONS:
- Adjust confidence_threshold for glazing specification accuracy
- Modify image_size_min for glazing detail visibility
- Change table_relevance_threshold for glazing specification tables
- Tune content_classification_threshold for glazing content recognition
""",
            
            'impact_rating': """
You are analyzing window impact rating extraction results. Focus on:

IMPACT RATING TYPES TO IDENTIFY:
- Small Missile Impact ratings
- Large Missile Impact ratings  
- Both Missile Impact ratings
- ASTM/AAMA test compliance
- Hurricane/storm ratings
- Impact resistance certifications

EVALUATION CRITERIA:
1. Are impact rating types (small/large/both missile) correctly identified?
2. Are impact test results and certifications captured?
3. Are compliance standards (ASTM, AAMA) referenced?
4. Is the extraction complete for all impact rating information?

PARAMETER RECOMMENDATIONS:
- Adjust confidence_threshold for impact rating accuracy
- Modify image_size_min for impact test result visibility  
- Change table_relevance_threshold for impact rating tables
- Tune content_classification_threshold for impact content recognition
""",
            
            'design_pressure': """
You are analyzing window design pressure extraction results. Focus on:

DESIGN PRESSURE DATA TO IDENTIFY:
- Design pressure ratings (DP values)
- Positive and negative pressure ratings
- Wind load specifications (PSF, PA, KPA)
- Structural load ratings
- Performance test results
- Pressure rating tables and charts

EVALUATION CRITERIA:
1. Are design pressure values and ratings accurately extracted?
2. Are pressure tables and performance data captured?
3. Are positive/negative pressure specifications identified?
4. Is the extraction comprehensive for all design pressure information?

PARAMETER RECOMMENDATIONS:
- Adjust confidence_threshold for pressure rating accuracy
- Modify image_size_min for pressure table visibility
- Change table_relevance_threshold for design pressure tables  
- Tune content_classification_threshold for pressure content recognition
"""
        }
    
    def analyze_extraction(self, document_id: str, debug: bool = False) -> Optional[Dict]:
        """Analyze characteristic-specific extraction results"""
        if not self.llm:
            print(f"❌ LLM not available for {self.characteristic} analysis")
            return None
        
        try:
            # Load extraction data
            extraction_file = f"feedback_data/{self.characteristic}_extraction_{document_id}.json"
            if not os.path.exists(extraction_file):
                print(f"❌ Extraction file not found: {extraction_file}")
                return None
            
            with open(extraction_file) as f:
                extraction_data = json.load(f)
            
            # Prepare analysis data
            analysis_data = self._prepare_analysis_data(extraction_data, debug)
            
            # Create analysis prompt
            system_prompt = self.analysis_prompts.get(self.characteristic, 
                                                    self.analysis_prompts['anchors'])
            
            user_prompt = f"""
CHARACTERISTIC FOCUS: {self.characteristic.upper()}

EXTRACTION RESULTS ANALYSIS:
{analysis_data}

Please analyze this {self.characteristic} extraction and provide:

1. EXTRACTION QUALITY ASSESSMENT (1-5 scale):
   - Accuracy: How well were {self.characteristic} items identified?
   - Completeness: Was all {self.characteristic} information captured?
   - Relevance: Are extracted items actually {self.characteristic}-related?

2. SPECIFIC {self.characteristic.upper()} FEEDBACK:
   - What {self.characteristic} information was found successfully?
   - What {self.characteristic} information was missed or incorrectly identified?
   - Are the characteristic-specific data points accurate?

3. PARAMETER RECOMMENDATIONS:
   - confidence_threshold: Currently {extraction_data.get('parameters_used', {}).get('confidence_threshold', 0.4)}
   - content_classification_threshold: Currently {extraction_data.get('parameters_used', {}).get('content_classification_threshold', 0.3)}
   - image_size_min: Currently {extraction_data.get('parameters_used', {}).get('image_size_min', 150)}
   - table_relevance_threshold: Currently {extraction_data.get('parameters_used', {}).get('table_relevance_threshold', 2)}

Suggest specific adjustments with reasoning for {self.characteristic} extraction improvement.

Respond in JSON format:
{{
    "extraction_quality": {{
        "accuracy": <1-5>,
        "completeness": <1-5>, 
        "relevance": <1-5>
    }},
    "characteristic_feedback": {{
        "found_successfully": ["item1", "item2"],
        "missed_or_incorrect": ["item3", "item4"],
        "data_accuracy": "assessment"
    }},
    "parameter_recommendations": {{
        "confidence_threshold": <new_value>,
        "content_classification_threshold": <new_value>,
        "image_size_min": <new_value>,
        "table_relevance_threshold": <new_value>,
        "reasoning": "explanation of changes"
    }}
}}
"""
            
            if debug:
                print(f"🤖 Sending {self.characteristic} analysis to LLM...")
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
                analysis_result = json.loads(response.content)
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
    
    def _prepare_analysis_data(self, extraction_data: Dict, debug: bool = False) -> str:
        """Prepare extraction data for LLM analysis"""
        sections = extraction_data.get('extracted_sections', [])
        analysis = extraction_data.get('extraction_analysis', {})
        params = extraction_data.get('parameters_used', {})
        
        # Summarize extraction results
        analysis_summary = []
        
        analysis_summary.append(f"EXTRACTION SUMMARY:")
        analysis_summary.append(f"- Total items: {len(sections)}")
        analysis_summary.append(f"- Average confidence: {analysis.get('avg_confidence', 0):.3f}")
        analysis_summary.append(f"- Coverage: {analysis.get('coverage', 'unknown')}")
        
        # Content type breakdown
        content_types = analysis.get('content_types', {})
        if content_types:
            analysis_summary.append(f"- Content types: {dict(content_types)}")
        
        # Characteristic-specific analysis
        char_specific = []
        if self.characteristic == 'anchors':
            anchor_count = analysis.get('anchor_types_identified', 0)
            char_specific.append(f"- Anchor types identified: {anchor_count}")
        elif self.characteristic == 'glazing':
            glazing_specs = analysis.get('glazing_specs_found', 0)
            char_specific.append(f"- Glazing specifications found: {glazing_specs}")
        elif self.characteristic == 'impact_rating':
            impact_ratings = analysis.get('impact_ratings_found', 0)
            char_specific.append(f"- Impact ratings found: {impact_ratings}")
        elif self.characteristic == 'design_pressure':
            pressure_values = analysis.get('pressure_values_found', 0)
            char_specific.append(f"- Pressure values found: {pressure_values}")
        
        if char_specific:
            analysis_summary.extend(char_specific)
        
        # Sample extractions (top 3 by confidence)
        top_sections = sorted(sections, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        
        analysis_summary.append(f"\nTOP EXTRACTIONS:")
        for i, section in enumerate(top_sections, 1):
            section_type = section.get('type', 'unknown')
            confidence = section.get('confidence', 0)
            page = section.get('page', 'unknown')
            
            analysis_summary.append(f"{i}. Type: {section_type}, Page: {page}, Confidence: {confidence:.3f}")
            
            # Add relevant content preview
            if 'table_analysis' in section:
                table_data = section['table_analysis'].get('data_points_found', [])
                if table_data:
                    analysis_summary.append(f"   Data points: {table_data[:3]}...")
            
            if 'characteristic_mentions' in section:
                mentions = section.get('characteristic_mentions', [])
                if mentions:
                    analysis_summary.append(f"   Mentions: {len(mentions)} relevant sentences")
            
            # Content preview
            content = section.get('content', '')
            if len(content) > 100:
                content_preview = content[:100] + "..."
            else:
                content_preview = content
            analysis_summary.append(f"   Content: {content_preview}")
        
        # Parameter status
        analysis_summary.append(f"\nCURRENT PARAMETERS:")
        for param, value in params.items():
            analysis_summary.append(f"- {param}: {value}")
        
        return '\n'.join(analysis_summary)
    
    def apply_recommendations(self, document_id: str, analysis_result: Dict, debug: bool = False) -> bool:
        """Apply LLM recommendations to parameters"""
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
                    from adaptive_agent import CharacteristicParameters
                    current_params = CharacteristicParameters().to_dict()
            except Exception as e:
                print(f"❌ Error loading current parameters: {e}")
                return False
            
            # Apply recommended changes
            changes_made = []
            for param_name, new_value in recommendations.items():
                if param_name == 'reasoning':
                    continue  # Skip reasoning text
                
                if param_name in current_params and new_value is not None:
                    try:
                        # Type conversion and validation
                        if param_name in ['image_size_min', 'table_relevance_threshold']:
                            new_value = max(50, min(500, int(float(new_value))))
                        elif param_name in ['confidence_threshold', 'content_classification_threshold']:
                            new_value = max(0.1, min(0.9, float(new_value)))
                        else:
                            new_value = float(new_value)
                        
                        old_value = current_params[param_name]
                        
                        # Only apply if change is significant
                        if abs(old_value - new_value) >= 0.01:
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
                
                # Log reasoning
                reasoning = recommendations.get('reasoning', 'No reasoning provided')
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
        quality = analysis_result.get('extraction_quality', {})
        char_feedback = analysis_result.get('characteristic_feedback', {})
        recommendations = analysis_result.get('parameter_recommendations', {})
        
        log_entry = {
            'document_id': document_id,
            'characteristic': self.characteristic,
            'timestamp': datetime.now().isoformat(),
            'quality_scores': {
                'accuracy': quality.get('accuracy', 0),
                'completeness': quality.get('completeness', 0),
                'relevance': quality.get('relevance', 0),
                'average': sum([quality.get('accuracy', 0), 
                              quality.get('completeness', 0), 
                              quality.get('relevance', 0)]) / 3
            },
            'characteristic_analysis': {
                'found_successfully': char_feedback.get('found_successfully', []),
                'missed_or_incorrect': char_feedback.get('missed_or_incorrect', []),
                'data_accuracy': char_feedback.get('data_accuracy', 'unknown')
            },
            'parameter_changes': changes_made,
            'llm_reasoning': recommendations.get('reasoning', ''),
            'llm_available': True,
            'llm_actually_used': True,
            'analysis_summary': f'LLM analysis for {self.characteristic} extraction'
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
            
            # Keep last 50 entries
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
            print(f"📝 Feedback logged to {log_file}")
            
        except Exception as e:
            print(f"⚠ Error saving feedback log: {e}")

def test_connection():
    """Test Azure OpenAI connection"""
    print("🔧 Testing Azure OpenAI connection...")
    
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please check your .env file")
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
        
        # Test with simple message
        messages = [HumanMessage(content="Reply with 'Connection successful' if you can read this.")]
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
    """Show recent feedback logs"""
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
                print(f"📊 {char_name.upper()} FEEDBACK HISTORY")
                print(f"{'='*60}")
                
                # Show last 5 entries
                recent_logs = logs[-5:]
                for i, log in enumerate(reversed(recent_logs), 1):
                    timestamp = datetime.fromisoformat(log.get('timestamp', '')).strftime('%Y-%m-%d %H:%M')
                    doc_id = log.get('document_id', 'unknown')
                    
                    quality = log.get('quality_scores', {})
                    avg_quality = quality.get('average', 0)
                    
                    changes = log.get('parameter_changes', [])
                    change_count = len(changes)
                    
                    print(f"\n{i}. {timestamp} | Doc: {doc_id[:8]}...")
                    print(f"   Quality: {avg_quality:.1f}/5 | Changes: {change_count}")
                    
                    if changes:
                        print(f"   Updates: {', '.join(changes[:2])}{'...' if len(changes) > 2 else ''}")
                    
                    reasoning = log.get('llm_reasoning', '')
                    if reasoning and len(reasoning) > 0:
                        print(f"   Reason: {reasoning[:80]}{'...' if len(reasoning) > 80 else ''}")
                
                print(f"\n📈 Total {char_name} feedback entries: {len(logs)}")
                
            except Exception as e:
                print(f"❌ Error reading {log_file}: {e}")
    
    if not found_logs:
        print("ℹ️  No feedback logs found. Run some extractions first!")

def main():
    parser = argparse.ArgumentParser(description="Characteristic-Specific LLM Feedback Analyzer")
    parser.add_argument("--analyze-characteristic", help="Analyze specific characteristic extraction")
    parser.add_argument("--test-connection", action="store_true", help="Test Azure OpenAI connection")
    parser.add_argument("--show-log", action="store_true", help="Show recent feedback logs")
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
    
    if args.analyze_characteristic and args.document_id:
        characteristic = args.analyze_characteristic
        document_id = args.document_id
        
        print(f"🤖 Starting {characteristic.upper()} LLM Analysis")
        print(f"📄 Document ID: {document_id}")
        
        if not LANGCHAIN_OK:
            print("❌ LangChain not available")
            return 1
        
        try:
            analyzer = CharacteristicLLMAnalyzer(characteristic)
            
            # Analyze extraction
            print(f"🔍 Analyzing {characteristic} extraction...")
            analysis_result = analyzer.analyze_extraction(document_id, args.debug)
            
            if not analysis_result:
                print(f"❌ Failed to analyze {characteristic} extraction")
                return 1
            
            print(f"✅ Analysis completed for {characteristic}")
            
            # Show quality scores
            quality = analysis_result.get('extraction_quality', {})
            print(f"\n📊 {characteristic.upper()} QUALITY SCORES:")
            print(f"   Accuracy: {quality.get('accuracy', 0)}/5")
            print(f"   Completeness: {quality.get('completeness', 0)}/5") 
            print(f"   Relevance: {quality.get('relevance', 0)}/5")
            
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
        print("  python llm_feedback.py --analyze-characteristic anchors doc123")
        print("  python llm_feedback.py --show-log --characteristic glazing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
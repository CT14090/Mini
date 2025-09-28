#!/usr/bin/env python3
# adaptive_agent.py
"""
Azure OpenAI-First Adaptive Agent - REDESIGNED VERSION
Intelligent document processing using Azure OpenAI Vision as the primary extraction engine
"""
import json
import os
import pathlib
import sys
import time
import hashlib
import argparse
import signal
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

# Timeout handler to prevent infinite processing
def timeout_handler(signum, frame):
    raise TimeoutError('Processing timed out - possible infinite loop detected')

# PDF to image conversion
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print("✓ PDF2Image available for conversion")
except ImportError:
    print("⚠ pdf2image not available - install with: pip install pdf2image")

# Azure OpenAI imports
AZURE_AVAILABLE = False
try:
    from langchain_openai import AzureChatOpenAI
    AZURE_AVAILABLE = True
    print("✓ Azure OpenAI available for vision analysis")
except ImportError:
    print("⚠ Azure OpenAI not available - limited functionality")

# Try to import Azure-first extractor
EXTRACTOR_AVAILABLE = False
try:
    from characteristic_based_extractor import AzureFirstExtractor
    EXTRACTOR_AVAILABLE = True
    print("✓ Azure-First Extractor available")
except ImportError:
    print("⚠ Azure-First extractor not available")

class AzureFirstDocumentProcessor:
    """Azure OpenAI-first document processor with integrated vision analysis"""
    
    def __init__(self):
        self.processing_timeout = 300  # 5 minutes max per characteristic
        self.max_pages_to_process = 20  # Reasonable limit for Azure API costs
        self.extractor = None
        
        if EXTRACTOR_AVAILABLE:
            try:
                self.extractor = AzureFirstExtractor()
                print("✓ Azure-First extractor initialized")
            except Exception as e:
                print(f"⚠ Error initializing extractor: {e}")
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check critical dependencies"""
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is required. Install with: pip install pdf2image")
        
        if not AZURE_AVAILABLE:
            print("⚠ Azure OpenAI not available - using fallback methods")
        
        # Check Azure OpenAI config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠ Azure OpenAI missing: {missing_vars}")
            print("  Will use fallback analysis methods")
        else:
            print("✓ Azure OpenAI configured for intelligent vision analysis")
    
    def process_document_for_characteristic(self, source: str, characteristic: str, debug: bool = False) -> str:
        """Process document using Azure OpenAI Vision as primary method"""
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.processing_timeout)
        
        try:
            start_time = time.time()
            doc_id = hashlib.md5(f"{source}_{characteristic}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            print(f"\n{'='*80}")
            print(f"🎯 AZURE OPENAI-FIRST CONTENT EXTRACTION")
            print(f"{'='*80}")
            print(f"📄 Source: {os.path.basename(source)}")
            print(f"🏗️ Target Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"🆔 Document ID: {doc_id}")
            print(f"⏰ Timeout: {self.processing_timeout}s")
            print(f"🤖 Method: Azure OpenAI Vision Analysis")
            print(f"{'='*80}")
            
            if not self.extractor:
                raise RuntimeError("Azure-First extractor not available")
            
            # Step 1: Convert PDF to images (optimized for Azure analysis)
            print(f"\n📖 STEP 1: CONVERTING PDF FOR AZURE ANALYSIS")
            print("─" * 50)
            pdf_images = self._convert_pdf_for_azure(source, debug)
            
            # Limit pages for cost control
            if len(pdf_images) > self.max_pages_to_process:
                print(f"⚠ Limiting to first {self.max_pages_to_process} pages (from {len(pdf_images)}) for cost control")
                pdf_images = pdf_images[:self.max_pages_to_process]
            
            print(f"✓ Processing {len(pdf_images)} pages with Azure OpenAI Vision")
            
            # Step 2: Azure Vision Analysis
            print(f"\n🤖 STEP 2: AZURE VISION ANALYSIS - {characteristic.upper()}")
            print("─" * 50)
            
            extracted_content = []
            pages_processed = 0
            azure_calls_made = 0
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Check if we're taking too long
                elapsed = time.time() - start_time
                if elapsed > self.processing_timeout * 0.8:  # 80% of timeout
                    print(f"⚠ Approaching timeout, stopping at page {page_num}")
                    break
                
                print(f"  📄 Processing page {page_num} with Azure Vision...")
                
                try:
                    # Extract content using Azure OpenAI Vision
                    page_content = self.extractor.extract_characteristic_content(
                        page_image, characteristic, page_num, debug
                    )
                    
                    # Track Azure API usage
                    if self.extractor.azure_client:
                        azure_calls_made += 1
                    
                    if page_content:
                        extracted_content.extend(page_content)
                        print(f"    ✓ Found {len(page_content)} relevant items")
                        
                        # Show extraction details
                        for item in page_content:
                            confidence = item.get('confidence', 0)
                            method = item.get('region_metadata', {}).get('detection_method', 'unknown')
                            print(f"      - {method}: confidence {confidence:.3f}")
                    else:
                        print(f"    - No relevant content found on page {page_num}")
                    
                    pages_processed += 1
                    
                    # Progress check
                    if pages_processed > 0 and pages_processed % 5 == 0:
                        print(f"  📊 Progress: {pages_processed}/{len(pdf_images)} pages, {len(extracted_content)} items extracted")
                        print(f"  🤖 Azure API calls made: {azure_calls_made}")
                
                except Exception as e:
                    print(f"    ❌ Error processing page {page_num}: {e}")
                    continue
                
                # Rate limiting for Azure API
                if azure_calls_made > 0:
                    time.sleep(1)  # 1 second between API calls
            
            # Step 3: Generate comprehensive results
            print(f"\n📋 STEP 3: GENERATING AZURE ANALYSIS RESULTS")
            print("─" * 50)
            
            processing_time = time.time() - start_time
            
            print(f"Azure Vision Analysis Results for {characteristic.replace('_', ' ').title()}:")
            print(f"  🔢 Total items extracted: {len(extracted_content)}")
            print(f"  📄 Pages processed: {pages_processed}/{len(pdf_images)}")
            print(f"  🤖 Azure API calls: {azure_calls_made}")
            print(f"  ⏱️ Processing time: {processing_time:.1f}s")
            print(f"  📈 Extraction rate: {len(extracted_content)/pages_processed:.1f} items/page" if pages_processed > 0 else "")
            
            # Analyze extraction methods
            if extracted_content:
                method_analysis = self._analyze_extraction_methods(extracted_content)
                print(f"  📊 Methods used: {method_analysis}")
                
                # Show average confidence
                avg_confidence = sum(item.get('confidence', 0) for item in extracted_content) / len(extracted_content)
                print(f"  🎯 Average confidence: {avg_confidence:.3f}")
                
                # Show Azure-specific insights
                azure_items = [item for item in extracted_content if 'azure' in item.get('region_metadata', {}).get('detection_method', '').lower()]
                if azure_items:
                    print(f"  🤖 Azure-analyzed items: {len(azure_items)}/{len(extracted_content)}")
            
            # Save results with Azure metadata
            extraction_data = self._create_azure_extraction_data(
                doc_id, source, characteristic, extracted_content, 
                processing_time, pages_processed, azure_calls_made
            )
            
            pathlib.Path("feedback_data").mkdir(exist_ok=True)
            extraction_file = f"feedback_data/extraction_{doc_id}.json"
            
            with open(extraction_file, 'w') as f:
                json.dump(extraction_data, f, indent=2)
            
            print(f"💾 Azure extraction results saved: {extraction_file}")
            
            # Step 4: Run integrated feedback analysis
            print(f"\n🔄 STEP 4: RUNNING INTEGRATED FEEDBACK ANALYSIS")
            print("─" * 50)
            
            self._run_integrated_feedback_safe(doc_id, debug)
            
            # Final summary
            print(f"\n{'='*80}")
            print(f"🎯 AZURE VISION EXTRACTION COMPLETED")
            print(f"{'='*80}")
            print(f"Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"Document ID: {doc_id}")
            print(f"Items extracted: {len(extracted_content)}")
            print(f"Azure API calls: {azure_calls_made}")
            print(f"Processing time: {processing_time:.1f}s")
            
            if extracted_content:
                high_conf = sum(1 for item in extracted_content if item.get('confidence', 0) > 0.7)
                print(f"High confidence items: {high_conf}/{len(extracted_content)}")
            
            print(f"View results: streamlit run feedback_interface.py")
            print(f"{'='*80}")
            
            return doc_id
            
        except TimeoutError:
            print(f"\n⚠ PROCESSING TIMED OUT after {self.processing_timeout}s")
            print("Consider reducing document size or increasing timeout.")
            return None
            
        except Exception as e:
            print(f"\n❌ PROCESSING FAILED: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return None
        
        finally:
            # Always clear the alarm
            signal.alarm(0)
    
    def _convert_pdf_for_azure(self, pdf_path: str, debug: bool = False) -> List:
        """Convert PDF to images optimized for Azure Vision analysis"""
        try:
            print(f"  Converting PDF for Azure analysis: {os.path.basename(pdf_path)}")
            
            images = convert_from_path(
                pdf_path,
                dpi=200,  # Optimal for Azure Vision (balance quality/cost)
                first_page=None,
                last_page=self.max_pages_to_process,
                fmt='RGB',
                thread_count=1
            )
            
            if debug:
                print(f"    Converted {len(images)} pages at 200 DPI for Azure Vision")
                for i, img in enumerate(images):
                    print(f"      Page {i+1}: {img.size[0]}x{img.size[1]} pixels")
            
            return images
            
        except Exception as e:
            print(f"❌ PDF conversion failed: {e}")
            raise
    
    def _analyze_extraction_methods(self, extracted_content: List[Dict]) -> str:
        """Analyze the methods used for extraction"""
        methods = {}
        
        for item in extracted_content:
            metadata = item.get('region_metadata', {})
            method = metadata.get('detection_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        # Format summary
        method_summary = []
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            if 'azure' in method.lower():
                method_summary.append(f"Azure({count})")
            else:
                method_summary.append(f"{method}({count})")
        
        return ', '.join(method_summary) if method_summary else 'none'
    
    def _create_azure_extraction_data(self, doc_id: str, source: str, characteristic: str, 
                                    content: List[Dict], processing_time: float, 
                                    pages_processed: int, azure_calls: int) -> Dict:
        """Create extraction data with Azure-specific metadata"""
        
        # Analyze extraction methods
        azure_items = []
        fallback_items = []
        total_confidence = 0
        
        for item in content:
            method = item.get('region_metadata', {}).get('detection_method', 'unknown')
            confidence = item.get('confidence', 0)
            total_confidence += confidence
            
            if 'azure' in method.lower():
                azure_items.append(item)
            else:
                fallback_items.append(item)
        
        avg_confidence = total_confidence / len(content) if content else 0
        high_confidence_count = sum(1 for item in content if item.get('confidence', 0) > 0.7)
        
        return {
            'document_id': doc_id,
            'document_path': source,
            'target_characteristic': characteristic,
            'extracted_sections': content,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'total_sections': len(content),
            'pages_processed': pages_processed,
            'processing_method': 'azure_openai_vision_first',
            'extraction_summary': {
                'total_items': len(content),
                'azure_analyzed_items': len(azure_items),
                'fallback_items': len(fallback_items),
                'azure_api_calls': azure_calls,
                'avg_confidence': avg_confidence,
                'high_confidence_items': high_confidence_count,
                'pages_with_content': list(set(item.get('page', 0) for item in content)),
                'extraction_rate_per_page': len(content) / pages_processed if pages_processed > 0 else 0
            },
            'azure_vision_metadata': {
                'azure_openai_enabled': azure_calls > 0,
                'training_data_integrated': True,
                'vision_analysis_method': 'azure_openai_gpt4_vision',
                'cost_optimized': True,
                'intelligent_region_selection': True,
                'contextual_understanding': True
            }
        }
    
    def _run_integrated_feedback_safe(self, doc_id: str, debug: bool = False):
        """Run integrated feedback analysis"""
        try:
            print("🔄 Running integrated feedback analysis...")
            
            # Use subprocess with timeout
            cmd = [sys.executable, "llm_feedback.py", "--analyze-and-apply", doc_id]
            if debug:
                cmd.append("--debug")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout for feedback
            )
            
            if result.returncode == 0:
                print("✅ Integrated feedback analysis completed")
                
                # Show feedback summary
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ['accuracy:', 'relevance:', 'parameter', 'applied']):
                            print(f"    {line}")
            else:
                print("⚠️ Feedback analysis had issues - continuing without it")
                if debug and result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print("⏰ Feedback analysis timed out - continuing")
        except Exception as e:
            if debug:
                print(f"❌ Feedback analysis error: {e}")
            else:
                print("⚠️ Feedback analysis unavailable")
    
    def process_all_characteristics(self, source: str, debug: bool = False) -> Dict[str, str]:
        """Process document for all characteristics using Azure Vision"""
        if not self.extractor:
            raise RuntimeError("Azure-First extractor not available")
        
        characteristics = self.extractor.get_available_characteristics()
        print(f"\n🔄 Processing document with Azure OpenAI Vision for ALL characteristics...")
        print(f"Available: {', '.join(characteristics)}")
        
        results = {}
        overall_start_time = time.time()
        total_azure_calls = 0
        
        for i, characteristic in enumerate(characteristics, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(characteristics)}: {characteristic}")
            print(f"{'='*60}")
            
            try:
                char_start_time = time.time()
                doc_id = self.process_document_for_characteristic(source, characteristic, debug)
                char_processing_time = time.time() - char_start_time
                
                if doc_id:
                    results[characteristic] = doc_id
                    print(f"✅ Completed: {characteristic} -> {doc_id} ({char_processing_time:.1f}s)")
                else:
                    results[characteristic] = None
                    print(f"❌ Failed: {characteristic} ({char_processing_time:.1f}s)")
                
            except Exception as e:
                print(f"❌ Error processing {characteristic}: {e}")
                results[characteristic] = None
        
        # Enhanced summary
        total_time = time.time() - overall_start_time
        successful = sum(1 for result in results.values() if result is not None)
        total = len(results)
        
        print(f"\n🎯 AZURE VISION BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{total} characteristics")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Average per characteristic: {total_time/total:.1f}s")
        print(f"Azure OpenAI method: Primary extraction engine")
        print(f"{'='*60}")
        
        for char, doc_id in results.items():
            status = "✅" if doc_id else "❌"
            print(f"{status} {char.replace('_', ' ').title()}: {doc_id or 'Failed'}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI-First Construction Document Agent")
    parser.add_argument("--source", help="PDF path or URL to process")
    parser.add_argument("--characteristic", help="Specific characteristic to extract")
    parser.add_argument("--all-characteristics", action="store_true", help="Extract all characteristics")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--list-characteristics", action="store_true", help="List available characteristics")
    parser.add_argument("--setup-labeled-data", action="store_true", help="Setup labeled data structure")
    parser.add_argument("--test-system", action="store_true", help="Test Azure system configuration")
    parser.add_argument("--test-azure", action="store_true", help="Test Azure OpenAI connection")
    
    args = parser.parse_args()
    
    if args.list_characteristics:
        if EXTRACTOR_AVAILABLE:
            try:
                extractor = AzureFirstExtractor()
                characteristics = extractor.get_available_characteristics()
                print("\n📋 Available Characteristics for Azure Vision Extraction:")
                for char in characteristics:
                    info = extractor.get_characteristic_info(char)
                    visual_indicators = info.get('visual_indicators', [])
                    print(f"  🏗️ {char.replace('_', ' ').title()}: {info.get('description', 'No description')}")
                    if visual_indicators:
                        print(f"     Looks for: {', '.join(visual_indicators[:2])}...")
            except Exception as e:
                print(f"❌ Error loading characteristics: {e}")
        else:
            print("❌ Azure-First extractor not available")
        return 0
    
    if args.setup_labeled_data:
        print("🏗️ Setting up labeled data structure for Azure Vision...")
        
        labeled_path = pathlib.Path("labeled_data")
        labeled_path.mkdir(exist_ok=True)
        
        # Enhanced characteristics with Azure-specific guidance
        characteristics_info = {
            "anchors": "Add clear technical drawings of anchor details, fastener assemblies, connection methods",
            "design_pressure": "Add pressure tables, wind load charts, performance data with clear numerical values", 
            "glazing": "Add glazing cross-sections, glass specifications, IGU details with clear visual elements",
            "impact_rating": "Add impact rating tables, test results, compliance charts with clear data"
        }
        
        for char, guidance in characteristics_info.items():
            char_path = labeled_path / char
            char_path.mkdir(exist_ok=True)
            
            readme_path = char_path / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"""Azure Vision Training Data: {char.replace('_', ' ').title()}

{guidance}

AZURE OPENAI VISION GUIDELINES:
- Azure OpenAI will analyze these examples to understand your target content
- Focus on clear, well-defined visual elements that represent {char.replace('_', ' ')}
- High-quality images (200+ DPI recommended)
- Clear, uncluttered examples that show the characteristic clearly
- 2-5 diverse examples per category (quality over quantity)

OPTIMAL CONTENT TYPES FOR AZURE VISION:
- Technical drawings with clear details and annotations
- Tables with structured data and clear headers
- Charts and graphs with visible data points
- Assembly details with clear visual elements
- Specification sheets with visual components

Azure OpenAI will:
✓ Understand the context and meaning of your examples
✓ Compare document content to these training examples
✓ Provide detailed reasoning for each extraction
✓ Focus on relevance and accuracy over quantity
""")
        
        print(f"✅ Azure Vision training structure created in: {labeled_path}")
        print("Add your training examples - Azure OpenAI will learn from these!")
        print("Focus on clear, representative examples of each characteristic.")
        return 0
    
    if args.test_azure:
        print("Testing Azure OpenAI connection...")
        
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain.schema import HumanMessage
            
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
            
            print(f"Endpoint: {endpoint[:30] + '...' if endpoint else 'Missing'}")
            print(f"API Key: {'✓ Present' if api_key else '❌ Missing'}")
            print(f"Deployment: {deployment if deployment else '❌ Missing'}")
            
            if not all([endpoint, api_key, deployment]):
                print("❌ Azure OpenAI credentials missing")
                print("\nRequired environment variables:")
                print("  - AZURE_OPENAI_ENDPOINT")
                print("  - AZURE_OPENAI_API_KEY")
                print("  - AZURE_OPENAI_DEPLOYMENT")
                return 1
            
            # Test connection
            client = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                azure_deployment=deployment,
                api_version="2024-02-01",
                temperature=0.1,
                max_tokens=100
            )
            
            test_message = HumanMessage(content="Hello, this is a test message.")
            
            print("Testing connection...")
            response = client.invoke([test_message])
            
            print("✅ Azure OpenAI connection successful")
            print(f"Response: {response.content[:50]}...")
            
            # Test vision capability if deployment supports it
            print("\nTesting vision capability...")
            vision_message = HumanMessage(content=[
                {"type": "text", "text": "Describe this simple test."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}}
            ])
            
            try:
                vision_response = client.invoke([vision_message])
                print("✅ Vision capability confirmed")
                print(f"Vision response: {vision_response.content[:50]}...")
            except Exception as ve:
                print(f"⚠️ Vision capability test failed: {ve}")
                print("This might be a deployment limitation, not a connection issue")
            
        except Exception as e:
            print(f"❌ Azure OpenAI test failed: {e}")
            return 1
        
        return 0
    
    if args.test_system:
        print("🧪 Testing Azure-First system configuration...")
        
        # Test imports
        print(f"✓ PDF2Image: {PDF2IMAGE_AVAILABLE}")
        print(f"✓ Azure OpenAI: {AZURE_AVAILABLE}")
        print(f"✓ Azure-First Extractor: {EXTRACTOR_AVAILABLE}")
        
        # Test Azure config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required_vars if not os.getenv(var)]
        print(f"✓ Azure OpenAI Config: {'Complete' if not missing else f'Missing {missing}'}")
        
        # Test training data
        labeled_path = pathlib.Path("labeled_data")
        if labeled_path.exists():
            categories = [d for d in labeled_path.iterdir() if d.is_dir()]
            total_images = 0
            
            print("Azure Vision Training Data:")
            for cat_dir in categories:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                total_images += len(images)
                
                if images:
                    print(f"  ✓ {cat_dir.name}: {len(images)} examples")
                else:
                    print(f"  ⚠ {cat_dir.name}: No examples - Azure needs training data!")
            
            print(f"✓ Total training examples: {total_images}")
            
            if total_images < 8:
                print("⚠ Consider adding more training examples (2+ per category)")
                print("  Azure OpenAI will learn better with diverse examples")
        else:
            print("⚠ No labeled_data directory - run --setup-labeled-data")
        
        print("\n🎯 Azure-First System Status:")
        if not missing and total_images >= 4 and AZURE_AVAILABLE:
            print("✅ System ready for Azure OpenAI-powered extraction")
        elif missing:
            print("❌ Configure Azure OpenAI credentials first")
        else:
            print("⚠ System partially ready - add training data for best results")
        
        return 0
    
    # Require source and characteristic selection
    if not args.source:
        parser.error("--source is required")
    
    if not args.characteristic and not args.all_characteristics:
        parser.error("Either --characteristic or --all-characteristics is required")
    
    print("🎯 Starting Azure OpenAI-First Document Processing...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        processor = AzureFirstDocumentProcessor()
        
        if args.all_characteristics:
            results = processor.process_all_characteristics(args.source, debug=args.debug)
            successful = sum(1 for result in results.values() if result is not None)
            print(f"\n🎉 AZURE VISION BATCH PROCESSING COMPLETE!")
            print(f"Successful: {successful}/{len(results)} characteristics")
            
            if successful > 0:
                print(f"\nNext steps:")
                print(f"  1. View results: streamlit run feedback_interface.py")
                print(f"  2. Run diagnostic: python diagnostic.py")
        else:
            doc_id = processor.process_document_for_characteristic(
                args.source, args.characteristic, debug=args.debug
            )
            
            if doc_id:
                print(f"\n🎉 AZURE VISION PROCESSING SUCCESSFUL!")
                print(f"Document ID: {doc_id}")
                print(f"Characteristic: {args.characteristic.replace('_', ' ').title()}")
                
                print(f"\nNext steps:")
                print(f"  1. View results: streamlit run feedback_interface.py")
                print(f"  2. Run diagnostic: python diagnostic.py")
            else:
                print(f"\n❌ AZURE VISION PROCESSING FAILED!")
                print(f"\nTroubleshooting:")
                print(f"  1. Test Azure connection: python adaptive_agent.py --test-azure")
                print(f"  2. Check training data: python adaptive_agent.py --setup-labeled-data")
                print(f"  3. Run with debug: python adaptive_agent.py --source {args.source} --characteristic {args.characteristic} --debug")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ PROCESSING FAILED: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
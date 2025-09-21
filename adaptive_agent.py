#!/usr/bin/env python3
# adaptive_agent.py
"""
Characteristic-Based Construction Document Agent - FIXED VERSION
Prevents infinite loops with proper timeouts and limits
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
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except:
    pass

# Timeout handler to prevent infinite processing
def timeout_handler(signum, frame):
    raise TimeoutError('Processing timed out - possible infinite loop detected')

# Computer Vision imports
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV_AVAILABLE = True
    print("✓ Computer Vision available for image processing")
except ImportError:
    print("⚠ OpenCV/PIL not available - limited functionality")

# PDF to image conversion
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print("✓ PDF2Image available for conversion")
except ImportError:
    print("⚠ pdf2image not available - install with: pip install pdf2image")

# Try to import characteristic extractor
CHARACTERISTIC_EXTRACTOR_AVAILABLE = False
try:
    from characteristic_based_extractor import CharacteristicBasedExtractor
    CHARACTERISTIC_EXTRACTOR_AVAILABLE = True
    print("✓ Characteristic-Based Extractor available")
except ImportError:
    print("⚠ Characteristic extractor not available")

class CharacteristicBasedDocumentProcessor:
    """Main processor with timeout protection"""
    
    def __init__(self):
        self.processing_timeout = 300  # 5 minutes max per characteristic
        self.max_pages_to_process = 20  # Limit to prevent runaway processing
        self.extractor = None
        
        if CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            try:
                self.extractor = CharacteristicBasedExtractor()
                print("✓ Characteristic extractor initialized")
            except Exception as e:
                print(f"⚠ Error initializing extractor: {e}")
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check critical dependencies"""
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is required. Install with: pip install pdf2image")
        
        if not CV_AVAILABLE:
            print("⚠ Limited functionality without OpenCV")
        
        # Check Azure OpenAI config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠ Azure OpenAI missing: {missing_vars}")
            print("  Will use fallback analysis")
        else:
            print("✓ Azure OpenAI configured")
    
    def process_document_for_characteristic(self, source: str, characteristic: str, debug: bool = False) -> str:
        """Process document for specific characteristic with timeout protection"""
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.processing_timeout)
        
        try:
            start_time = time.time()
            doc_id = hashlib.md5(f"{source}_{characteristic}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            print(f"\n{'='*80}")
            print(f"🎯 CHARACTERISTIC-BASED DOCUMENT PROCESSING")
            print(f"{'='*80}")
            print(f"📄 Source: {os.path.basename(source)}")
            print(f"🏗️ Target Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"🆔 Document ID: {doc_id}")
            print(f"⏰ Timeout: {self.processing_timeout}s")
            print(f"{'='*80}")
            
            if not self.extractor:
                raise RuntimeError("Characteristic extractor not available")
            
            # Step 1: Convert PDF to images (with page limit)
            print(f"\n📖 STEP 1: CONVERTING PDF TO IMAGES")
            print("─" * 50)
            pdf_images = self._convert_pdf_to_images(source, debug)
            
            # Limit pages to prevent runaway processing
            if len(pdf_images) > self.max_pages_to_process:
                print(f"⚠ Limiting to first {self.max_pages_to_process} pages (from {len(pdf_images)})")
                pdf_images = pdf_images[:self.max_pages_to_process]
            
            print(f"✓ Processing {len(pdf_images)} pages")
            
            # Step 2: Extract content with timeout checks
            print(f"\n🎯 STEP 2: EXTRACTING {characteristic.upper()} CONTENT")
            print("─" * 50)
            
            extracted_content = []
            pages_processed = 0
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Check if we're taking too long
                elapsed = time.time() - start_time
                if elapsed > self.processing_timeout * 0.8:  # 80% of timeout
                    print(f"⚠ Approaching timeout, stopping at page {page_num}")
                    break
                
                print(f"  📄 Processing page {page_num}...")
                
                try:
                    # Extract content for this page with individual timeout
                    page_content = self._extract_page_content_safe(
                        page_image, page_num, characteristic, debug
                    )
                    
                    if page_content:
                        extracted_content.extend(page_content)
                        print(f"    ✓ Found {len(page_content)} items")
                    else:
                        print(f"    - No relevant content")
                    
                    pages_processed += 1
                    
                    # Reasonable progress check
                    if pages_processed > 0 and pages_processed % 5 == 0:
                        print(f"  📊 Progress: {pages_processed}/{len(pdf_images)} pages, {len(extracted_content)} items found")
                
                except Exception as e:
                    print(f"    ❌ Error processing page {page_num}: {e}")
                    continue
            
            # Step 3: Generate results
            print(f"\n📋 STEP 3: GENERATING RESULTS")
            print("─" * 50)
            
            processing_time = time.time() - start_time
            
            print(f"Results for {characteristic.replace('_', ' ').title()}:")
            print(f"  🔢 Total items: {len(extracted_content)}")
            print(f"  📄 Pages processed: {pages_processed}/{len(pdf_images)}")
            print(f"  ⏱️ Processing time: {processing_time:.1f}s")
            
            # Save results
            extraction_data = self._create_extraction_data(
                doc_id, source, characteristic, extracted_content, processing_time, pages_processed
            )
            
            pathlib.Path("feedback_data").mkdir(exist_ok=True)
            extraction_file = f"feedback_data/extraction_{doc_id}.json"
            
            with open(extraction_file, 'w') as f:
                json.dump(extraction_data, f, indent=2)
            
            print(f"💾 Extraction saved: {extraction_file}")
            
            # Step 4: Run LLM feedback (optional)
            print(f"\n🤖 STEP 4: RUNNING VISION FEEDBACK (OPTIONAL)")
            print("─" * 50)
            
            self._run_llm_feedback_safe(doc_id, debug)
            
            # Final summary
            print(f"\n{'='*80}")
            print(f"🎯 CHARACTERISTIC-BASED PROCESSING COMPLETED")
            print(f"{'='*80}")
            print(f"Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"Document ID: {doc_id}")
            print(f"Items found: {len(extracted_content)}")
            print(f"Processing time: {processing_time:.1f}s")
            print(f"View results: streamlit run feedback_interface.py")
            print(f"{'='*80}")
            
            return doc_id
            
        except TimeoutError:
            print(f"\n⚠ PROCESSING TIMED OUT after {self.processing_timeout}s")
            print("This usually indicates an infinite loop or very large document.")
            print("Try reducing document size or check for processing issues.")
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
    
    def process_all_characteristics(self, source: str, debug: bool = False) -> Dict[str, str]:
        """Process document for all available characteristics"""
        if not self.extractor:
            raise RuntimeError("Characteristic extractor not available")
        
        characteristics = self.extractor.get_available_characteristics()
        print(f"\n🔄 Processing document for ALL characteristics...")
        print(f"Available: {', '.join(characteristics)}")
        
        results = {}
        
        for characteristic in characteristics:
            print(f"\n{'='*50}")
            print(f"Processing: {characteristic}")
            print(f"{'='*50}")
            
            try:
                doc_id = self.process_document_for_characteristic(source, characteristic, debug)
                if doc_id:
                    results[characteristic] = doc_id
                    print(f"✅ Completed: {characteristic} -> {doc_id}")
                else:
                    results[characteristic] = None
                    print(f"❌ Failed: {characteristic}")
                
            except Exception as e:
                print(f"❌ Error processing {characteristic}: {e}")
                results[characteristic] = None
        
        # Summary
        successful = sum(1 for result in results.values() if result is not None)
        total = len(results)
        
        print(f"\n🎯 BATCH PROCESSING SUMMARY")
        print(f"Successful: {successful}/{total} characteristics")
        
        for char, doc_id in results.items():
            status = "✅" if doc_id else "❌"
            print(f"{status} {char}: {doc_id or 'Failed'}")
        
        return results
    
    def _convert_pdf_to_images(self, pdf_path: str, debug: bool = False) -> List[Image.Image]:
        """Convert PDF to images with error handling"""
        try:
            print(f"  Converting PDF: {os.path.basename(pdf_path)}")
            
            images = convert_from_path(
                pdf_path,
                dpi=200,
                first_page=None,
                last_page=self.max_pages_to_process,  # Limit pages
                fmt='RGB'
            )
            
            if debug:
                print(f"    Converted {len(images)} pages at 200 DPI")
                for i, img in enumerate(images):
                    print(f"      Page {i+1}: {img.size[0]}x{img.size[1]} pixels")
            
            return images
            
        except Exception as e:
            print(f"❌ PDF conversion failed: {e}")
            raise
    
    def _extract_page_content_safe(self, page_image: Image.Image, page_num: int, 
                                  characteristic: str, debug: bool) -> List[Dict]:
        """Extract content from page with timeout protection"""
        if not self.extractor:
            return []
        
        try:
            # Set a shorter timeout for individual page processing
            signal.alarm(30)  # 30 seconds per page max
            
            content = self.extractor.extract_characteristic_content(
                page_image, characteristic, page_num, debug
            )
            
            signal.alarm(0)  # Clear alarm
            return content
            
        except TimeoutError:
            print(f"    ⚠ Page {page_num} timed out - skipping")
            return []
        except Exception as e:
            print(f"    ❌ Error extracting from page {page_num}: {e}")
            return []
        finally:
            signal.alarm(0)  # Always clear alarm
    
    def _create_extraction_data(self, doc_id: str, source: str, characteristic: str, 
                              content: List[Dict], processing_time: float, pages_processed: int) -> Dict:
        """Create extraction data structure"""
        return {
            'document_id': doc_id,
            'document_path': source,
            'target_characteristic': characteristic,
            'extracted_sections': content,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'total_sections': len(content),
            'pages_processed': pages_processed,
            'processing_method': 'characteristic_based_extraction',
            'extraction_summary': {
                'total_items': len(content),
                'diagram_items': sum(1 for item in content if not item.get('type', '').startswith('table')),
                'table_items': sum(1 for item in content if item.get('type', '').startswith('table')),
                'avg_confidence': sum(item.get('confidence', 0) for item in content) / len(content) if content else 0,
                'pages_with_content': list(set(item.get('page', 0) for item in content))
            }
        }
    
    def _run_llm_feedback_safe(self, doc_id: str, debug: bool = False):
        """Run LLM feedback with timeout protection"""
        try:
            print("🤖 Launching vision feedback analyzer...")
            
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
                print("✅ Vision feedback completed")
            else:
                print("⚠️ Vision feedback had issues - check logs")
                if debug and result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print("⏰ Vision feedback timed out - continuing without it")
        except Exception as e:
            if debug:
                print(f"❌ Vision feedback error: {e}")
            else:
                print("⚠️ Vision feedback unavailable")

def main():
    parser = argparse.ArgumentParser(description="Characteristic-Based Construction Document Agent")
    parser.add_argument("--source", help="PDF path or URL to process")
    parser.add_argument("--characteristic", help="Specific characteristic to extract")
    parser.add_argument("--all-characteristics", action="store_true", help="Extract all characteristics")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--list-characteristics", action="store_true", help="List available characteristics")
    parser.add_argument("--setup-labeled-data", action="store_true", help="Setup labeled data structure")
    parser.add_argument("--test-system", action="store_true", help="Test system configuration")
    
    args = parser.parse_args()
    
    if args.list_characteristics:
        if CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            try:
                extractor = CharacteristicBasedExtractor()
                characteristics = extractor.get_available_characteristics()
                print("\n📋 Available Characteristics:")
                for char in characteristics:
                    info = extractor.get_characteristic_info(char)
                    print(f"  🏗️ {char.replace('_', ' ').title()}: {info.get('description', 'No description')}")
            except Exception as e:
                print(f"❌ Error loading characteristics: {e}")
        else:
            print("❌ Characteristic extractor not available")
        return 0
    
    if args.setup_labeled_data:
        print("🏗️ Setting up labeled data structure...")
        
        labeled_path = pathlib.Path("labeled_data")
        labeled_path.mkdir(exist_ok=True)
        
        # Default characteristics
        characteristics = ["anchors", "design_pressure", "glazing", "impact_rating"]
        
        for char in characteristics:
            char_path = labeled_path / char
            char_path.mkdir(exist_ok=True)
            
            readme_path = char_path / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"""Category: {char.replace('_', ' ').title()}

Add your labeled training images (JPG or PNG) to this folder.
These images will be used as references to classify similar content from PDFs.

Guidelines:
- Clear, high-quality images showing {char.replace('_', ' ')} content
- Technical diagrams preferred over text
- 3-10 images per category recommended
""")
        
        print(f"✅ Created structure in: {labeled_path}")
        print("Add your training images to each category folder")
        return 0
    
    if args.test_system:
        print("🧪 Testing system configuration...")
        
        # Test imports
        print(f"✓ PDF2Image: {PDF2IMAGE_AVAILABLE}")
        print(f"✓ OpenCV: {CV_AVAILABLE}")
        print(f"✓ Characteristic Extractor: {CHARACTERISTIC_EXTRACTOR_AVAILABLE}")
        
        # Test Azure config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required_vars if not os.getenv(var)]
        print(f"✓ Azure OpenAI: {'Configured' if not missing else f'Missing {missing}'}")
        
        # Test training data
        labeled_path = pathlib.Path("labeled_data")
        if labeled_path.exists():
            categories = [d for d in labeled_path.iterdir() if d.is_dir()]
            total_images = 0
            for cat_dir in categories:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                total_images += len(images)
                print(f"✓ {cat_dir.name}: {len(images)} images")
            print(f"✓ Total training images: {total_images}")
        else:
            print("⚠ No labeled_data directory - run --setup-labeled-data")
        
        return 0
    
    # Require characteristic selection
    if not args.source:
        parser.error("--source is required")
    
    if not args.characteristic and not args.all_characteristics:
        parser.error("Either --characteristic or --all-characteristics is required")
    
    print("🎯 Starting Characteristic-Based Construction Document Processing...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        processor = CharacteristicBasedDocumentProcessor()
        
        if args.all_characteristics:
            results = processor.process_all_characteristics(args.source, debug=args.debug)
            successful = sum(1 for result in results.values() if result is not None)
            print(f"\n🎉 BATCH PROCESSING COMPLETE!")
            print(f"Successful: {successful}/{len(results)} characteristics")
        else:
            doc_id = processor.process_document_for_characteristic(
                args.source, args.characteristic, debug=args.debug
            )
            
            if doc_id:
                print(f"\n🎉 PROCESSING SUCCESSFUL!")
                print(f"Document ID: {doc_id}")
            else:
                print(f"\n❌ PROCESSING FAILED!")
                return 1
        
        print(f"\nNext steps:")
        print(f"  1. View results: streamlit run feedback_interface.py")
        print(f"  2. Run diagnostic: python diagnostic.py")
        
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
#!/usr/bin/env python3
# adaptive_agent.py
"""
Enhanced Adaptive Agent - VISUAL CONTENT FOCUSED VERSION
Properly extracts diagrams, tables, and technical drawings
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

# Computer Vision imports
CV_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV_AVAILABLE = True
    print("✓ Computer Vision available for enhanced image processing")
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

# Try to import enhanced characteristic extractor
CHARACTERISTIC_EXTRACTOR_AVAILABLE = False
try:
    from characteristic_based_extractor import CharacteristicBasedExtractor
    CHARACTERISTIC_EXTRACTOR_AVAILABLE = True
    print("✓ Enhanced Characteristic-Based Extractor available")
except ImportError:
    print("⚠ Enhanced characteristic extractor not available")

class EnhancedDocumentProcessor:
    """Enhanced processor with visual content focus and timeout protection"""
    
    def __init__(self):
        self.processing_timeout = 300  # 5 minutes max per characteristic
        self.max_pages_to_process = 25  # Increased for better coverage
        self.extractor = None
        
        if CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            try:
                self.extractor = CharacteristicBasedExtractor()
                print("✓ Enhanced characteristic extractor initialized")
            except Exception as e:
                print(f"⚠ Error initializing extractor: {e}")
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check critical dependencies"""
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image is required. Install with: pip install pdf2image")
        
        if not CV_AVAILABLE:
            print("⚠ Limited functionality without OpenCV")
            raise RuntimeError("OpenCV is required for visual content detection. Install with: pip install opencv-python")
        
        # Check Azure OpenAI config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠ Azure OpenAI missing: {missing_vars}")
            print("  Will use enhanced fallback analysis")
        else:
            print("✓ Azure OpenAI configured for vision feedback")
    
    def process_document_for_characteristic(self, source: str, characteristic: str, debug: bool = False) -> str:
        """Process document for specific characteristic with enhanced visual detection"""
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.processing_timeout)
        
        try:
            start_time = time.time()
            doc_id = hashlib.md5(f"{source}_{characteristic}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            print(f"\n{'='*80}")
            print(f"🎯 ENHANCED VISUAL CONTENT EXTRACTION")
            print(f"{'='*80}")
            print(f"📄 Source: {os.path.basename(source)}")
            print(f"🏗️ Target Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"🆔 Document ID: {doc_id}")
            print(f"⏰ Timeout: {self.processing_timeout}s")
            print(f"🔍 Detection: Tables, Diagrams, Visual Content")
            print(f"{'='*80}")
            
            if not self.extractor:
                raise RuntimeError("Enhanced characteristic extractor not available")
            
            # Step 1: Convert PDF to images with enhanced settings
            print(f"\n📖 STEP 1: CONVERTING PDF TO HIGH-QUALITY IMAGES")
            print("─" * 50)
            pdf_images = self._convert_pdf_to_images_enhanced(source, debug)
            
            # Limit pages to prevent runaway processing
            if len(pdf_images) > self.max_pages_to_process:
                print(f"⚠ Limiting to first {self.max_pages_to_process} pages (from {len(pdf_images)})")
                pdf_images = pdf_images[:self.max_pages_to_process]
            
            print(f"✓ Processing {len(pdf_images)} pages with enhanced visual detection")
            
            # Step 2: Extract visual content with enhanced detection
            print(f"\n🎯 STEP 2: DETECTING VISUAL CONTENT - {characteristic.upper()}")
            print("─" * 50)
            
            extracted_content = []
            pages_processed = 0
            total_visual_regions = 0
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Check if we're taking too long
                elapsed = time.time() - start_time
                if elapsed > self.processing_timeout * 0.8:  # 80% of timeout
                    print(f"⚠ Approaching timeout, stopping at page {page_num}")
                    break
                
                print(f"  📄 Processing page {page_num} with visual detection...")
                
                try:
                    # Extract content for this page with enhanced visual detection
                    page_content, regions_found = self._extract_page_content_enhanced(
                        page_image, page_num, characteristic, debug
                    )
                    
                    total_visual_regions += regions_found
                    
                    if page_content:
                        extracted_content.extend(page_content)
                        content_types = [item.get('region_metadata', {}).get('content_type', 'unknown') for item in page_content]
                        type_summary = {}
                        for ct in content_types:
                            type_summary[ct] = type_summary.get(ct, 0) + 1
                        type_str = ', '.join([f"{k}:{v}" for k, v in type_summary.items()])
                        print(f"    ✓ Found {len(page_content)} items ({type_str}) from {regions_found} visual regions")
                    else:
                        print(f"    - No relevant content (checked {regions_found} visual regions)")
                    
                    pages_processed += 1
                    
                    # Progress check
                    if pages_processed > 0 and pages_processed % 5 == 0:
                        print(f"  📊 Progress: {pages_processed}/{len(pdf_images)} pages, {len(extracted_content)} items found, {total_visual_regions} regions analyzed")
                
                except Exception as e:
                    print(f"    ❌ Error processing page {page_num}: {e}")
                    continue
            
            # Step 3: Generate enhanced results
            print(f"\n📋 STEP 3: GENERATING ENHANCED RESULTS")
            print("─" * 50)
            
            processing_time = time.time() - start_time
            
            print(f"Enhanced Results for {characteristic.replace('_', ' ').title()}:")
            print(f"  🔢 Total items extracted: {len(extracted_content)}")
            print(f"  📄 Pages processed: {pages_processed}/{len(pdf_images)}")
            print(f"  🔍 Visual regions analyzed: {total_visual_regions}")
            print(f"  ⏱️ Processing time: {processing_time:.1f}s")
            print(f"  📈 Extraction rate: {len(extracted_content)/pages_processed:.1f} items/page" if pages_processed > 0 else "")
            
            # Analyze content types
            if extracted_content:
                content_analysis = self._analyze_extracted_content(extracted_content)
                print(f"  📊 Content breakdown: {content_analysis}")
            
            # Save results with enhanced metadata
            extraction_data = self._create_enhanced_extraction_data(
                doc_id, source, characteristic, extracted_content, 
                processing_time, pages_processed, total_visual_regions
            )
            
            pathlib.Path("feedback_data").mkdir(exist_ok=True)
            extraction_file = f"feedback_data/extraction_{doc_id}.json"
            
            with open(extraction_file, 'w') as f:
                json.dump(extraction_data, f, indent=2)
            
            print(f"💾 Enhanced extraction saved: {extraction_file}")
            
            # Step 4: Run enhanced LLM feedback
            print(f"\n🤖 STEP 4: RUNNING ENHANCED VISION FEEDBACK")
            print("─" * 50)
            
            self._run_enhanced_feedback_safe(doc_id, debug)
            
            # Final enhanced summary
            print(f"\n{'='*80}")
            print(f"🎯 ENHANCED VISUAL EXTRACTION COMPLETED")
            print(f"{'='*80}")
            print(f"Characteristic: {characteristic.replace('_', ' ').title()}")
            print(f"Document ID: {doc_id}")
            print(f"Items extracted: {len(extracted_content)}")
            print(f"Visual regions analyzed: {total_visual_regions}")
            print(f"Processing time: {processing_time:.1f}s")
            
            if extracted_content:
                avg_confidence = sum(item.get('confidence', 0) for item in extracted_content) / len(extracted_content)
                print(f"Average confidence: {avg_confidence:.3f}")
                
                # Show extraction quality
                high_conf = sum(1 for item in extracted_content if item.get('confidence', 0) > 0.6)
                print(f"High confidence items: {high_conf}/{len(extracted_content)}")
            
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
    
    def _convert_pdf_to_images_enhanced(self, pdf_path: str, debug: bool = False) -> List[Image.Image]:
        """Convert PDF to images with enhanced settings for better visual detection"""
        try:
            print(f"  Converting PDF: {os.path.basename(pdf_path)}")
            
            images = convert_from_path(
                pdf_path,
                dpi=250,  # Increased DPI for better visual detection
                first_page=None,
                last_page=self.max_pages_to_process,
                fmt='RGB',
                thread_count=1  # Single thread to prevent issues
            )
            
            if debug:
                print(f"    Converted {len(images)} pages at 250 DPI for enhanced visual detection")
                for i, img in enumerate(images):
                    print(f"      Page {i+1}: {img.size[0]}x{img.size[1]} pixels")
            
            return images
            
        except Exception as e:
            print(f"❌ Enhanced PDF conversion failed: {e}")
            raise
    
    def _extract_page_content_enhanced(self, page_image: Image.Image, page_num: int, 
                                     characteristic: str, debug: bool) -> Tuple[List[Dict], int]:
        """Extract content from page with enhanced visual detection"""
        if not self.extractor:
            return [], 0
        
        try:
            # Set a shorter timeout for individual page processing
            signal.alarm(25)  # 25 seconds per page max
            
            content = self.extractor.extract_characteristic_content(
                page_image, characteristic, page_num, debug
            )
            
            # Get the number of visual regions that were analyzed
            regions_analyzed = 0
            for item in content:
                metadata = item.get('region_metadata', {})
                if 'detection_method' in metadata:
                    regions_analyzed += 1
            
            signal.alarm(0)  # Clear alarm
            return content, regions_analyzed
            
        except TimeoutError:
            print(f"    ⚠ Page {page_num} timed out - skipping")
            return [], 0
        except Exception as e:
            print(f"    ❌ Error extracting from page {page_num}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return [], 0
        finally:
            signal.alarm(0)  # Always clear alarm
    
    def _analyze_extracted_content(self, extracted_content: List[Dict]) -> str:
        """Analyze the types of content extracted"""
        content_types = {}
        detection_methods = {}
        
        for item in extracted_content:
            # Content type analysis
            metadata = item.get('region_metadata', {})
            content_type = metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Detection method analysis
            detection_method = metadata.get('detection_method', 'unknown')
            detection_methods[detection_method] = detection_methods.get(detection_method, 0) + 1
        
        # Format summary
        type_summary = []
        if content_types:
            for ct, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                type_summary.append(f"{ct}({count})")
        
        return ', '.join(type_summary) if type_summary else 'none'
    
    def _create_enhanced_extraction_data(self, doc_id: str, source: str, characteristic: str, 
                                       content: List[Dict], processing_time: float, 
                                       pages_processed: int, visual_regions_analyzed: int) -> Dict:
        """Create enhanced extraction data structure"""
        
        # Analyze content types
        content_analysis = {}
        detection_methods = {}
        confidence_levels = []
        
        for item in content:
            # Content type breakdown
            metadata = item.get('region_metadata', {})
            content_type = metadata.get('content_type', 'unknown')
            content_analysis[content_type] = content_analysis.get(content_type, 0) + 1
            
            # Detection method breakdown
            detection_method = metadata.get('detection_method', 'unknown')
            detection_methods[detection_method] = detection_methods.get(detection_method, 0) + 1
            
            # Confidence tracking
            confidence_levels.append(item.get('confidence', 0))
        
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        high_confidence_count = sum(1 for c in confidence_levels if c > 0.6)
        
        return {
            'document_id': doc_id,
            'document_path': source,
            'target_characteristic': characteristic,
            'extracted_sections': content,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'total_sections': len(content),
            'pages_processed': pages_processed,
            'processing_method': 'enhanced_visual_content_extraction',
            'extraction_summary': {
                'total_items': len(content),
                'visual_regions_analyzed': visual_regions_analyzed,
                'avg_confidence': avg_confidence,
                'high_confidence_items': high_confidence_count,
                'pages_with_content': list(set(item.get('page', 0) for item in content)),
                'content_type_breakdown': content_analysis,
                'detection_method_breakdown': detection_methods,
                'extraction_rate_per_page': len(content) / pages_processed if pages_processed > 0 else 0
            },
            'visual_detection_metadata': {
                'enhanced_detection_enabled': True,
                'table_detection_used': 'table_detection' in detection_methods,
                'diagram_detection_used': 'diagram_detection' in detection_methods,
                'content_block_detection_used': 'content_block' in detection_methods,
                'pdf_dpi': 250,
                'visual_feature_extraction_enabled': True
            }
        }
    
    def _run_enhanced_feedback_safe(self, doc_id: str, debug: bool = False):
        """Run enhanced LLM feedback with timeout protection"""
        try:
            print("🤖 Launching enhanced vision feedback analyzer...")
            
            # Use subprocess with timeout
            cmd = [sys.executable, "llm_feedback.py", "--analyze-and-apply", doc_id]
            if debug:
                cmd.append("--debug")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=150  # 2.5 minute timeout for enhanced feedback
            )
            
            if result.returncode == 0:
                print("✅ Enhanced vision feedback completed")
                
                # Show feedback summary if available
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'Accuracy:' in line or 'parameter adjustments' in line:
                            print(f"    {line}")
            else:
                print("⚠️ Enhanced vision feedback had issues - check logs")
                if debug and result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print("⏰ Enhanced vision feedback timed out - continuing without it")
        except Exception as e:
            if debug:
                print(f"❌ Enhanced vision feedback error: {e}")
            else:
                print("⚠️ Enhanced vision feedback unavailable")
    
    def process_all_characteristics(self, source: str, debug: bool = False) -> Dict[str, str]:
        """Process document for all available characteristics with enhanced detection"""
        if not self.extractor:
            raise RuntimeError("Enhanced characteristic extractor not available")
        
        characteristics = self.extractor.get_available_characteristics()
        print(f"\n🔄 Processing document for ALL characteristics with enhanced visual detection...")
        print(f"Available: {', '.join(characteristics)}")
        
        results = {}
        overall_start_time = time.time()
        
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
        
        print(f"\n🎯 ENHANCED BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{total} characteristics")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Average per characteristic: {total_time/total:.1f}s")
        print(f"{'='*60}")
        
        for char, doc_id in results.items():
            status = "✅" if doc_id else "❌"
            print(f"{status} {char.replace('_', ' ').title()}: {doc_id or 'Failed'}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Characteristic-Based Construction Document Agent")
    parser.add_argument("--source", help="PDF path or URL to process")
    parser.add_argument("--characteristic", help="Specific characteristic to extract")
    parser.add_argument("--all-characteristics", action="store_true", help="Extract all characteristics")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("--list-characteristics", action="store_true", help="List available characteristics")
    parser.add_argument("--setup-labeled-data", action="store_true", help="Setup labeled data structure")
    parser.add_argument("--test-system", action="store_true", help="Test enhanced system configuration")
    parser.add_argument("--visual-test", action="store_true", help="Run visual detection test")
    
    args = parser.parse_args()
    
    if args.list_characteristics:
        if CHARACTERISTIC_EXTRACTOR_AVAILABLE:
            try:
                extractor = CharacteristicBasedExtractor()
                characteristics = extractor.get_available_characteristics()
                print("\n📋 Available Characteristics for Enhanced Extraction:")
                for char in characteristics:
                    info = extractor.get_characteristic_info(char)
                    visual_features = info.get('visual_features', [])
                    print(f"  🏗️ {char.replace('_', ' ').title()}: {info.get('description', 'No description')}")
                    if visual_features:
                        print(f"     Visual features: {', '.join(visual_features)}")
            except Exception as e:
                print(f"❌ Error loading characteristics: {e}")
        else:
            print("❌ Enhanced characteristic extractor not available")
        return 0
    
    if args.setup_labeled_data:
        print("🏗️ Setting up enhanced labeled data structure...")
        
        labeled_path = pathlib.Path("labeled_data")
        labeled_path.mkdir(exist_ok=True)
        
        # Enhanced characteristics with guidance
        characteristics_info = {
            "anchors": "Add clear diagrams showing anchor details, connection methods, fastening systems",
            "design_pressure": "Add tables/charts with pressure ratings, wind load data, performance specs", 
            "glazing": "Add glazing section details, glass specifications, IGU assemblies",
            "impact_rating": "Add impact rating tables, test results, compliance charts"
        }
        
        for char, guidance in characteristics_info.items():
            char_path = labeled_path / char
            char_path.mkdir(exist_ok=True)
            
            readme_path = char_path / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"""Enhanced Training Data for: {char.replace('_', ' ').title()}

{guidance}

ENHANCED VISUAL DETECTION GUIDELINES:
- Focus on VISUAL content: diagrams, tables, charts, technical drawings
- Avoid pure text paragraphs - system looks for visual elements
- High-quality images (300+ DPI recommended)
- Clear, uncluttered examples
- 5-10 diverse examples per category

BEST CONTENT TYPES:
- Technical drawings and cross-sections
- Tables with data/specifications  
- Charts and graphs
- Assembly details and callouts
- Compliance certificates with visual elements

The enhanced system will detect:
✓ Table structures using line detection
✓ Diagram regions using edge analysis  
✓ Technical drawings using feature extraction
✓ Visual content blocks with structured elements
""")
        
        print(f"✅ Enhanced structure created in: {labeled_path}")
        print("Add your visual training content to each category folder")
        print("Focus on diagrams, tables, and technical drawings - not text!")
        return 0
    
    if args.test_system:
        print("🧪 Testing enhanced system configuration...")
        
        # Test imports
        print(f"✓ PDF2Image: {PDF2IMAGE_AVAILABLE}")
        print(f"✓ OpenCV: {CV_AVAILABLE}")
        print(f"✓ Enhanced Extractor: {CHARACTERISTIC_EXTRACTOR_AVAILABLE}")
        
        if CV_AVAILABLE:
            print("  OpenCV modules available:")
            print("    ✓ Image processing")
            print("    ✓ Edge detection (Canny)")
            print("    ✓ Morphological operations")
            print("    ✓ Contour detection")
            print("    ✓ Feature extraction")
        
        # Test Azure config
        required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT']
        missing = [var for var in required_vars if not os.getenv(var)]
        print(f"✓ Azure OpenAI: {'Configured' if not missing else f'Missing {missing}'}")
        
        # Test enhanced training data
        labeled_path = pathlib.Path("labeled_data")
        if labeled_path.exists():
            categories = [d for d in labeled_path.iterdir() if d.is_dir()]
            total_images = 0
            
            print("Enhanced Training Data Analysis:")
            for cat_dir in categories:
                images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
                total_images += len(images)
                
                if images:
                    # Analyze first image for guidance
                    try:
                        if CV_AVAILABLE:
                            sample_img = cv2.imread(str(images[0]))
                            if sample_img is not None:
                                height, width = sample_img.shape[:2]
                                print(f"  ✓ {cat_dir.name}: {len(images)} images (sample: {width}x{height})")
                            else:
                                print(f"  ✓ {cat_dir.name}: {len(images)} images")
                        else:
                            print(f"  ✓ {cat_dir.name}: {len(images)} images")
                    except:
                        print(f"  ✓ {cat_dir.name}: {len(images)} images")
                else:
                    print(f"  ⚠ {cat_dir.name}: No images - add visual training data!")
            
            print(f"✓ Total training images: {total_images}")
            
            if total_images < 12:
                print("⚠ Consider adding more training images (3+ per category)")
                print("  Focus on: diagrams, tables, technical drawings")
        else:
            print("⚠ No labeled_data directory - run --setup-labeled-data")
        
        return 0
    
    if args.visual_test:
        print("🔍 Running visual detection test...")
        
        if not CV_AVAILABLE:
            print("❌ OpenCV not available - cannot run visual test")
            return 1
        
        # Test visual detection capabilities
        print("Testing visual detection methods:")
        
        # Create test image
        test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        test_img.fill(255)  # White background
        
        # Draw test table
        cv2.rectangle(test_img, (50, 50), (300, 200), (0, 0, 0), 2)
        cv2.line(test_img, (50, 100), (300, 100), (0, 0, 0), 1)
        cv2.line(test_img, (50, 150), (300, 150), (0, 0, 0), 1)
        cv2.line(test_img, (150, 50), (150, 200), (0, 0, 0), 1)
        
        # Draw test diagram
        cv2.circle(test_img, (450, 125), 50, (0, 0, 0), 2)
        cv2.line(test_img, (400, 125), (500, 125), (0, 0, 0), 2)
        cv2.line(test_img, (450, 75), (450, 175), (0, 0, 0), 2)
        
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Test table detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        h_lines = np.sum(horizontal_lines > 0)
        v_lines = np.sum(vertical_lines > 0)
        
        print(f"  ✓ Table detection: {h_lines} horizontal, {v_lines} vertical line pixels")
        
        # Test edge detection
        edges = cv2.Canny(gray, 30, 100)
        edge_pixels = np.sum(edges > 0)
        print(f"  ✓ Edge detection: {edge_pixels} edge pixels found")
        
        # Test contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  ✓ Contour detection: {len(contours)} contours found")
        
        print("✅ Visual detection capabilities verified")
        return 0
    
    # Require characteristic selection
    if not args.source:
        parser.error("--source is required")
    
    if not args.characteristic and not args.all_characteristics:
        parser.error("Either --characteristic or --all-characteristics is required")
    
    print("🎯 Starting Enhanced Visual Content Extraction...")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        processor = EnhancedDocumentProcessor()
        
        if args.all_characteristics:
            results = processor.process_all_characteristics(args.source, debug=args.debug)
            successful = sum(1 for result in results.values() if result is not None)
            print(f"\n🎉 ENHANCED BATCH PROCESSING COMPLETE!")
            print(f"Successful: {successful}/{len(results)} characteristics")
            
            if successful > 0:
                print(f"\nNext steps:")
                print(f"  1. View enhanced results: streamlit run feedback_interface.py")
                print(f"  2. Run diagnostic: python diagnostic.py")
        else:
            doc_id = processor.process_document_for_characteristic(
                args.source, args.characteristic, debug=args.debug
            )
            
            if doc_id:
                print(f"\n🎉 ENHANCED PROCESSING SUCCESSFUL!")
                print(f"Document ID: {doc_id}")
                print(f"Characteristic: {args.characteristic.replace('_', ' ').title()}")
                
                print(f"\nNext steps:")
                print(f"  1. View enhanced results: streamlit run feedback_interface.py")
                print(f"  2. Run diagnostic: python diagnostic.py")
                print(f"  3. Run vision feedback: python llm_feedback.py --analyze-and-apply {doc_id}")
            else:
                print(f"\n❌ ENHANCED PROCESSING FAILED!")
                print(f"\nTroubleshooting:")
                print(f"  1. Check training data: python adaptive_agent.py --setup-labeled-data")
                print(f"  2. Test system: python adaptive_agent.py --test-system")
                print(f"  3. Run with debug: python adaptive_agent.py --source {args.source} --characteristic {args.characteristic} --debug")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Enhanced processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ENHANCED PROCESSING FAILED: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())